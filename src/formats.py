"""Shared structured-output serializers and parsers for JSON and key-value formats."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

Slot = Dict[str, str]
Parsed = Dict[str, Any]


def slots_to_multiset(slots: List[Slot]) -> Counter:
    """Normalize slots to a multiset of (name, value) pairs (case-sensitive values)."""
    return Counter((s["name"], s["value"]) for s in slots)


def to_structured_output(intent: str, slots: List[Slot], fmt: str) -> str:
    """Serialize gold intent/slots into the requested structured-output format."""
    if fmt == "json":
        obj = {
            "intent": intent,
            "slots": [{"name": s["name"], "value": s["value"]} for s in slots],
        }
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if fmt == "key_value":
        if not slots:
            slots_str = "NONE"
        else:
            slots_str = " | ".join(f"{s['name']}:{s['value']}" for s in slots)
        return f"intent={intent}\nslots={slots_str}"
    raise ValueError(f"Unknown format: {fmt}")


def _extract_json_object(text: str) -> Optional[str]:
    """Find the first top-level JSON object in text, if any."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_slots(raw_slots: Any) -> Optional[List[Slot]]:
    if raw_slots is None:
        return []
    if not isinstance(raw_slots, list):
        return None
    slots: List[Slot] = []
    for item in raw_slots:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            return None
        slots.append({"name": name, "value": value})
    return slots


def parse_json_prediction(text: str) -> Optional[Parsed]:
    candidate = _extract_json_object(text.strip())
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    intent = obj.get("intent")
    if not isinstance(intent, str):
        return None
    slots = _normalize_slots(obj.get("slots", []))
    if slots is None:
        return None
    # Reject unexpected top-level keys beyond intent/slots (schema check soft: allow only these)
    extra = set(obj.keys()) - {"intent", "slots"}
    if extra:
        return None
    return {"intent": intent, "slots": slots}


def parse_key_value_prediction(text: str) -> Optional[Parsed]:
    cleaned = text.strip()
    # Drop common markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    intent_match = re.search(r"(?im)^\s*intent\s*=\s*(.+?)\s*$", cleaned)
    slots_match = re.search(r"(?im)^\s*slots\s*=\s*(.+?)\s*$", cleaned)
    if not intent_match or not slots_match:
        return None

    intent = intent_match.group(1).strip()
    slots_raw = slots_match.group(1).strip()
    if not intent:
        return None

    if slots_raw.upper() == "NONE" or slots_raw == "":
        return {"intent": intent, "slots": []}

    slots: List[Slot] = []
    parts = [p.strip() for p in slots_raw.split("|")]
    for part in parts:
        if not part:
            continue
        if ":" not in part:
            return None
        name, value = part.split(":", 1)
        name, value = name.strip(), value.strip()
        if not name:
            return None
        slots.append({"name": name, "value": value})
    return {"intent": intent, "slots": slots}


def parse_prediction(text: str, fmt: str) -> Optional[Parsed]:
    if fmt == "json":
        return parse_json_prediction(text)
    if fmt == "key_value":
        return parse_key_value_prediction(text)
    raise ValueError(f"Unknown format: {fmt}")


def is_format_valid(text: str, fmt: str) -> bool:
    return parse_prediction(text, fmt) is not None


def slot_f1_counts(
    gold_slots: List[Slot], pred_slots: List[Slot]
) -> Tuple[int, int, int]:
    """Return (tp, pred_count, gold_count) for micro slot F1 over (name, value) pairs."""
    gold = slots_to_multiset(gold_slots)
    pred = slots_to_multiset(pred_slots)
    tp = sum((gold & pred).values())
    return tp, sum(pred.values()), sum(gold.values())


def exact_match(gold: Parsed, pred: Optional[Parsed]) -> bool:
    if pred is None:
        return False
    if gold["intent"] != pred["intent"]:
        return False
    return slots_to_multiset(gold["slots"]) == slots_to_multiset(pred["slots"])
