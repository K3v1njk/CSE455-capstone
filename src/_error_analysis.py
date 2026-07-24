#!/usr/bin/env python3
"""Write results/base/error_analysis.md from locked-format test predictions."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import sys

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "base"
sys.path.insert(0, str(ROOT / "src"))
from formats import exact_match  # noqa: E402


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def classify(row: dict) -> str:
    gold = row["gold"]
    parsed = row.get("parsed_prediction")
    if parsed is None:
        return "malformed"
    if exact_match(gold, parsed):
        return "correct"
    if parsed.get("intent") != gold.get("intent"):
        return "wrong_intent"
    return "wrong_slot"


def fmt_example(row: dict) -> str:
    gold = row["gold"]
    parsed = row.get("parsed_prediction")
    raw = (row.get("raw_prediction") or "").strip()
    if len(raw) > 400:
        raw = raw[:400] + "…"
    return (
        f"- **id**: `{row['id']}`\n"
        f"  - request: {row.get('request','')}\n"
        f"  - gold: `{json.dumps(gold, ensure_ascii=False)}`\n"
        f"  - parsed: `{json.dumps(parsed, ensure_ascii=False) if parsed is not None else None}`\n"
        f"  - raw: `{raw}`\n"
    )


def analyze_dataset(ds: str, fmt: str, n_per: int = 5) -> str:
    path = RES / f"{ds}_test_predictions.jsonl"
    rows = load_jsonl(path)
    buckets = {"correct": [], "wrong_intent": [], "wrong_slot": [], "malformed": []}
    for row in rows:
        buckets[classify(row)].append(row)
    counts = {k: len(v) for k, v in buckets.items()}
    parts = [f"## {ds.upper()} (format=`{fmt}`, N={len(rows)})", "", f"Counts: {counts}", ""]
    for kind in ("correct", "wrong_intent", "wrong_slot", "malformed"):
        parts.append(f"### {kind.replace('_', ' ').title()} (~{n_per})")
        parts.append("")
        for row in buckets[kind][:n_per]:
            parts.append(fmt_example(row))
        if not buckets[kind]:
            parts.append("_None in this category._\n")
    # Failure pattern notes
    invented = 0
    prose = 0
    for row in buckets["malformed"] + buckets["wrong_slot"] + buckets["wrong_intent"]:
        raw = (row.get("raw_prediction") or "")
        if raw.strip().startswith("{") or "intent=" in raw:
            pass
        else:
            prose += 1
        parsed = row.get("parsed_prediction")
        if parsed:
            # invented intents rough check not available without label list here
            pass
    parts += [
        "### Failure pattern notes",
        "",
        f"- Malformed / unparsable: {counts['malformed']} ({100*counts['malformed']/max(len(rows),1):.1f}%)",
        f"- Wrong intent (parsable): {counts['wrong_intent']}",
        f"- Wrong slots only: {counts['wrong_slot']}",
        f"- Correct exact match: {counts['correct']}",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    fmt = (RES / "selected_format.txt").read_text().strip()
    body = [
        "# Error analysis (base Qwen, locked test format)",
        "",
        f"Locked format: `{fmt}`",
        "",
        "Categories: correct exact-match, wrong intent, wrong slots (intent OK), malformed/unparsable.",
        "",
    ]
    for ds in ("snips", "atis"):
        body.append(analyze_dataset(ds, fmt))
        body.append("")
    out = RES / "error_analysis.md"
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
