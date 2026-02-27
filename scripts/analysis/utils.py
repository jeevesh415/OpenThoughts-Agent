"""Shared utilities for analysis scripts."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]

_EPISODE_PATTERN = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# JSONL iteration
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield parsed dicts from a JSONL file, raising on malformed lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number}: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Conversation text extraction
# ---------------------------------------------------------------------------

def _extract_from_message_content(content) -> str:
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            parts.append(text)
    elif isinstance(content, list):
        for item in content:
            parts.append(_extract_from_message_content(item))
    return "\n".join(p for p in parts if p)


def extract_conversation_text(record) -> str:
    """Extract the full concatenated text from a conversation record.

    Handles ``messages``, ``conversations``, and various fallback fields.
    """
    if isinstance(record, dict):
        messages = record.get("messages") or record.get("conversations")
        if isinstance(messages, list):
            collected: list[str] = []
            for message in messages:
                if isinstance(message, dict):
                    if "content" in message:
                        collected.append(_extract_from_message_content(message["content"]))
                    for key in ("value", "text"):
                        value = message.get(key)
                        if isinstance(value, str):
                            collected.append(value)
                elif isinstance(message, str):
                    collected.append(message)
            combined = "\n".join(chunk for chunk in collected if chunk)
            if combined:
                return combined
        for field in ("conversation", "text", "prompt", "content"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(record, ensure_ascii=False)


def count_turns(record) -> int:
    """Estimate the number of turns (messages) in a record."""
    if isinstance(record, dict):
        messages = record.get("messages") or record.get("conversations")
        if isinstance(messages, list):
            return len(messages)
        turn_count = record.get("turn_count")
        if isinstance(turn_count, int):
            return turn_count
    return 0


# ---------------------------------------------------------------------------
# Episode number extraction
# ---------------------------------------------------------------------------

def extract_episode_numbers(values: Iterable) -> List[int]:
    """Parse integer episode indices from various formats (str, int, dict)."""
    episodes: List[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            episodes.append(int(value))
            continue
        if isinstance(value, str):
            cleaned = value.replace("-", " ").replace("_", " ")
            match = _EPISODE_PATTERN.search(cleaned)
            if match:
                episodes.append(int(match.group(1)))
            continue
        if isinstance(value, dict):
            inner = value.get("episode")
            if isinstance(inner, (int, float)):
                episodes.append(int(inner))
            elif isinstance(inner, str):
                cleaned_inner = inner.replace("-", " ").replace("_", " ")
                match = _EPISODE_PATTERN.search(cleaned_inner)
                if match:
                    episodes.append(int(match.group(1)))
    return episodes


# ---------------------------------------------------------------------------
# Token counting (tiktoken)
# ---------------------------------------------------------------------------

def get_tiktoken_encoder():
    """Return a tiktoken encoder, or ``None`` if tiktoken is unavailable."""
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    """Count tokens in *text* using *encoder*, falling back to whitespace split."""
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text.split())


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------

def load_hf_trace_dataset(repo_id: str, split: str = "train"):
    """Load a HuggingFace dataset with a helpful error message on failure."""
    from datasets import load_dataset
    try:
        return load_dataset(repo_id, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{repo_id}' (split={split}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Reward / result extraction
# ---------------------------------------------------------------------------

def extract_reward(record) -> Optional[float]:
    """Extract a numeric reward from a record's ``result`` field.

    Returns ``None`` if the value is missing, null-like, or non-numeric.
    """
    if isinstance(record, dict):
        candidate = record.get("result")
    else:
        candidate = record
    if candidate is None:
        return None
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped or stripped.lower() in ("none", "null", ""):
            return None
        try:
            return float(stripped)
        except (TypeError, ValueError):
            return None
    return None


def extract_error_type(record) -> Optional[str]:
    """Extract an error type name from a record's ``result`` field.

    Returns the string value when it is not parseable as a number (i.e. it's
    an exception class name like ``"AgentTimeoutError"``).  Returns ``None``
    for numeric results or missing values.
    """
    if isinstance(record, dict):
        candidate = record.get("result")
    else:
        candidate = record
    if candidate is None:
        return None
    if isinstance(candidate, (int, float)):
        return None
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped or stripped.lower() in ("none", "null"):
            return None
        try:
            float(stripped)
            return None  # it's a number, not an error
        except (TypeError, ValueError):
            return stripped
    return None


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def extract_date(record) -> Optional[datetime]:
    """Parse the ``date`` field of a record into a :class:`datetime`.

    Accepts ISO 8601 strings.  Returns ``None`` on failure.
    """
    if isinstance(record, dict):
        candidate = record.get("date")
    else:
        candidate = record
    if candidate is None:
        return None
    if isinstance(candidate, datetime):
        return candidate
    if isinstance(candidate, str):
        try:
            return datetime.fromisoformat(candidate)
        except (TypeError, ValueError):
            return None
    return None
