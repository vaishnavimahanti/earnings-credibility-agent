"""
Thin LLM wrapper so the rest of the codebase doesn't care which provider we use.
Always returns Pydantic objects via structured outputs.
"""

from __future__ import annotations
import json
import re
import time
from typing import Callable, Type, TypeVar
from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


def _is_transient_llm_error(error: Exception) -> bool:
    """Best-effort detection for retryable provider/network failures."""
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    name = type(error).__name__.lower()
    transient_names = (
        "apiconnectionerror",
        "apitimeouterror",
        "internalservererror",
        "ratelimiterror",
        "serviceunavailableerror",
        "timeout",
    )
    return any(marker in name for marker in transient_names)


def _parse_structured_json(text: str, schema: Type[T]) -> T:
    """Parse provider JSON, repairing common harmless formatting slips."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    # Some providers occasionally return a trailing comma before the closing
    # brace/bracket. That is invalid JSON but unambiguous, so repair it once
    # before handing the text to Pydantic.
    repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        try:
            return schema.model_validate_json(repaired)
        except ValidationError as err:
            # The most common schema miss after JSON repair is a verbose text
            # field that exceeds a max_length guard. Trim those fields and
            # validate once more so a single overlong explanation does not
            # abort a long evaluation run.
            if not any(e.get("type") == "string_too_long" for e in err.errors()):
                raise
            data = json.loads(repaired)
            if isinstance(data, dict):
                for e in err.errors():
                    if e.get("type") != "string_too_long":
                        continue
                    loc = e.get("loc") or ()
                    if len(loc) != 1:
                        continue
                    field = loc[0]
                    max_len = (e.get("ctx") or {}).get("max_length")
                    if max_len and isinstance(data.get(field), str):
                        data[field] = data[field][:max_len].rstrip()
            return schema.model_validate(data)


class LLMClient:
    """Single-interface client. Use .structured() for typed outputs."""

    def __init__(self, model: str | None = None):
        self.provider = config.get_llm_provider()
        self.model = model or config.LLM_MODEL

        if self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> T:
        """Return an instance of `schema` from the LLM."""
        if self.provider == "anthropic":
            return self._with_retries(
                lambda: self._anthropic_structured(prompt, schema, system, max_tokens)
            )
        return self._with_retries(
            lambda: self._openai_structured(prompt, schema, system, max_tokens)
        )

    def _with_retries(self, call: Callable[[], T], attempts: int = 4) -> T:
        """Retry transient provider failures, but not schema validation errors."""
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return call()
            except Exception as e:
                if not _is_transient_llm_error(e) or attempt == attempts:
                    raise
                last_error = e
                time.sleep(delay)
                delay *= 2
        raise last_error  # pragma: no cover

    def _anthropic_structured(
        self, prompt: str, schema: Type[T], system: str | None, max_tokens: int
    ) -> T:
        json_schema = schema.model_json_schema()
        sys_prompt = (system or "") + (
            "\n\nYou must respond ONLY with a JSON object matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}\n\n"
            "Do not include markdown fences, prose, or anything outside the JSON."
        )

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=sys_prompt.strip(),
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_structured_json(resp.content[0].text, schema)

    def _openai_structured(
        self, prompt: str, schema: Type[T], system: str | None, max_tokens: int
    ) -> T:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.parsed


_default_client: LLMClient | None = None


def default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
