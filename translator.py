"""Ask a Databricks-served LLM to play one of the teachers in `teachers.py`."""

from __future__ import annotations

import logging
import random
import re
import textwrap
import time
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from teachers import Teacher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Model:
    name: str  # the request's `model` field
    route: str  # path under the workspace host that serves it


# Listed in the sidebar in this order; the first one is the default. To add a model, copy
# `model` and the end of `base_url` from the "Query" snippet on its Serving page.
MODELS = {
    "Qwen 3.5 122B · serving endpoint": Model("databricks-qwen35-122b-a10b", "serving-endpoints"),
    "Qwen 3.5 122B · AI Gateway": Model("system.ai.qwen35-122b-a10b", "ai-gateway/mlflow/v1"),
}
DEFAULT_MODEL = next(iter(MODELS.values()))
# Transient failures worth retrying; BadRequest/AuthError are not - retrying just burns time.
RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


class TranslationError(RuntimeError):
    """Raised when the model returns nothing usable."""


@dataclass(frozen=True)
class Answer:
    text: str
    truncated: bool  # the model hit max_tokens, so the text stops mid-way


def _get_client(route: str) -> OpenAI:
    """Build the OpenAI-compatible client.

    In a notebook the token came from dbutils; a Databricks App has no notebook context, so
    the SDK resolves it instead - service principal OAuth in the app, CLI profile locally.
    Not cached: OAuth tokens expire, and rebuilding this per call is cheap.
    """
    cfg = WorkspaceClient().config
    token = cfg.authenticate()["Authorization"].removeprefix("Bearer ")
    return OpenAI(api_key=token, base_url=f"{cfg.host}/{route}")


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


def translate(
    source: str,
    teacher: Teacher,
    note: str | None = None,
    *,
    model: Model = DEFAULT_MODEL,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Answer:
    """Hand `source` to the model playing `teacher`.

    Returns Python source or a Markdown report, depending on `teacher.output`. Nothing is
    executed. `max_tokens` defaults to the teacher's own budget.
    """
    source = textwrap.dedent(source).strip()
    if not source:
        raise ValueError("source is empty")

    prompt = f"{teacher.request}\n\n```{teacher.fence}\n{source}\n```"
    if note:
        prompt += f"\n\nAdditional note from me: {note}"

    client = _get_client(model.route)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model.name,
                max_tokens=max_tokens or teacher.max_tokens,
                temperature=temperature,  # deterministic output makes diffs reviewable
                messages=[
                    {"role": "system", "content": teacher.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            choice = response.choices[0]
            text = _extract_text(choice.message)
            text = _strip_code_fences(text) if teacher.output == "python" else text.strip()
            truncated = choice.finish_reason == "length"
            if not text:
                # Reasoning comes out of the same budget, and can use all of it.
                hint = " - max tokens ran out while it was still reasoning" if truncated else ""
                raise TranslationError(f"Model returned an empty response{hint}")
            return Answer(text, truncated)
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
