"""Spark SQL -> PySpark DataFrame API translator backed by a Databricks-served LLM."""

from __future__ import annotations

import logging
import os
import random
import re
import textwrap
import time

from databricks.sdk import WorkspaceClient
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("SERVING_MODEL", "system.ai.qwen35-122b-a10b")
DEFAULT_MAX_TOKENS = 16_000
# Transient failures worth retrying; BadRequest/AuthError are not - retrying just burns time.
RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

SYSTEM_PROMPT = """You are a PySpark expert teaching someone who is moving over from SQL.
You rewrite Spark SQL queries into the DataFrame API (PySpark).

Rules:
- Return ONLY Python code, with no ``` fences and no introductory or closing commentary.
- Assign the result to a variable named `df`. Do not call .show(), .display() or .collect().
- Load tables via spark.table("name").
- Use `from pyspark.sql import functions as F` and write F.col(...) rather than strings when it reads better.
- Above every non-trivial step, add a one-line `#` comment explaining which part of the SQL
  it corresponds to (e.g. `# HAVING -> filter after aggregation`).
- Idiomatic PySpark, not a literal transcription of the SQL.
- Add some tips and tricks if there is something worth mentioning"""

_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


class TranslationError(RuntimeError):
    """Raised when the model returns nothing usable."""


def _get_client() -> OpenAI:
    """Build the OpenAI-compatible client.

    In a notebook the token came from dbutils; a Databricks App has no notebook context, so
    the SDK resolves it instead - service principal OAuth in the app, CLI profile locally.
    Not cached: OAuth tokens expire, and rebuilding this per call is cheap.
    """
    cfg = WorkspaceClient().config
    token = cfg.authenticate()["Authorization"].removeprefix("Bearer ")
    return OpenAI(api_key=token, base_url=f"{cfg.host}/ai-gateway/mlflow/v1")


def _extract_text(message) -> str:
    """Normalise the response payload: some gateways return a string, others a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    raise TranslationError(f"Unexpected content type: {type(content)!r}")


def _strip_code_fences(text: str) -> str:
    """The prompt forbids fences, but models add them anyway - strip defensively."""
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


def translate_sql_to_pyspark(
    query: str,
    note: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Translate a Spark SQL query into PySpark DataFrame API code.

    Returns the generated Python source as a string. Does not execute it.
    """
    query = textwrap.dedent(query).strip()
    if not query:
        raise ValueError("query is empty")

    prompt = f"Rewrite this Spark SQL query in PySpark:\n\n```sql\n{query}\n```"
    if note:
        prompt += f"\n\nAdditional note from me: {note}"

    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,  # deterministic output makes diffs reviewable
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            code = _strip_code_fences(_extract_text(response.choices[0].message))
            if not code:
                raise TranslationError("Model returned an empty response")
            return code
        except RETRYABLE_ERRORS as exc:
            last_error = exc
            backoff = 2**attempt + random.uniform(0, 0.5)  # jitter avoids retry stampedes
            logger.warning(
                "LLM call failed (attempt %d/%d): %s - retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                backoff,
            )
            time.sleep(backoff)

    raise TranslationError(f"Translation failed after {max_retries} attempts") from last_error
