"""Spark SQL -> PySpark DataFrame API translator backed by a Databricks-served LLM."""

from __future__ import annotations

import logging
import os
import random
import textwrap
import time
from collections.abc import Iterator
from functools import lru_cache

from databricks.sdk import WorkspaceClient
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# The serving-endpoint name, not the UC model path (`system.ai.*` gives ENDPOINT_NOT_FOUND here).
# In a Databricks App this comes from app.yaml; locally it falls back to the default.
DEFAULT_MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-qwen35-122b-a10b")
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


class TranslationError(RuntimeError):
    """Raised when the model returns nothing usable."""


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Build the OpenAI-compatible client once per process.

    The SDK resolves credentials from the environment: inside a Databricks App those are the
    service principal's OAuth vars, locally it is whatever `databricks auth login` wrote.
    """
    return WorkspaceClient().serving_endpoints.get_open_ai_client()


def strip_code_fences(text: str) -> str:
    """The prompt forbids fences, but models add them anyway - strip defensively.

    Tolerates an unterminated opening fence so this also works on a partial stream.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _build_messages(query: str, note: str | None) -> list[dict[str, str]]:
    query = textwrap.dedent(query).strip()
    if not query:
        raise ValueError("query is empty")

    prompt = f"Rewrite this Spark SQL query in PySpark:\n\n```sql\n{query}\n```"
    if note:
        prompt += f"\n\nAdditional note from me: {note}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def stream_translation(
    query: str,
    note: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Iterator[str]:
    """Yield the generated PySpark source in chunks as the model produces it.

    Chunks are raw model output; run the accumulated text through `strip_code_fences`
    before showing or saving it.
    """
    messages = _build_messages(query, note)
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        emitted = False
        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,  # deterministic output makes diffs reviewable
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    emitted = True
                    yield delta
            if not emitted:
                raise TranslationError("Model returned an empty response")
            return
        except RETRYABLE_ERRORS as exc:
            # Once the caller has seen output we cannot restart without duplicating text.
            if emitted:
                raise
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


def translate_sql_to_pyspark(query: str, note: str | None = None, **kwargs) -> str:
    """Translate a Spark SQL query into PySpark DataFrame API code.

    Returns the generated Python source as a string. Does not execute it.
    """
    code = strip_code_fences("".join(stream_translation(query, note, **kwargs)))
    if not code:
        raise TranslationError("Model returned an empty response")
    return code
