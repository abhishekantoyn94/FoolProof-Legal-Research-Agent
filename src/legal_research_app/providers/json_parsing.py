"""Shared robust JSON-from-LLM parsing. Every structured LLM call in this app
(Synonimise, the research agent's planner/synthesizer/challenger) goes through
this rather than assuming the model obeys a "respond with only JSON"
instruction -- empirically it doesn't always (see README's Phase 4 finding).
"""

from __future__ import annotations

import json

from legal_research_app.logging_setup import get_logger

logger = get_logger("json_parsing")


class LLMJSONParseError(ValueError):
    pass


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def parse_json_object(content: str, *, context: str = "") -> dict:
    """Parses a JSON object from LLM output, tolerating markdown fences and
    surrounding prose. Raises LLMJSONParseError (never silently returns {})
    if nothing usable is found -- callers must decide how to degrade."""
    text = _strip_code_fence(content)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON object from LLM response (%s): %r", context, text[:200])
    raise LLMJSONParseError(f"Could not parse a JSON object from the model's response ({context}).")
