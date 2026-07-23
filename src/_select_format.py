#!/usr/bin/env python3
"""Build format_comparison.md and print selected format from validation metrics."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "base"


def load(ds: str, fmt: str) -> dict:
    return json.loads((RES / f"{ds}_validation_{fmt}_metrics.json").read_text())


def main() -> None:
    rows = []
    for ds in ("snips", "atis"):
        for fmt in ("json", "key_value"):
            m = load(ds, fmt)
            rows.append((ds, fmt, m))

    by_fmt = {"json": [], "key_value": []}
    for ds, fmt, m in rows:
        by_fmt[fmt].append(m)

    means = {}
    for fmt, ms in by_fmt.items():
        means[fmt] = {
            "exact_match": sum(m["exact_match"] for m in ms) / len(ms),
            "format_valid_pct": sum(m["format_valid_pct"] for m in ms) / len(ms),
            "slot_f1": sum(m["slot_f1"] for m in ms) / len(ms),
        }

    # Selection: mean EM, then valid%, then slot F1
    selected = max(
        ("json", "key_value"),
        key=lambda f: (
            means[f]["exact_match"],
            means[f]["format_valid_pct"],
            means[f]["slot_f1"],
        ),
    )

    lines = [
        "# Format comparison (validation)",
        "",
        "Frozen model: `Qwen/Qwen2.5-0.5B-Instruct` (`configs/base_qwen.yaml`).",
        "Selection rule: higher mean exact-match across SNIPS+ATIS; ties broken by format-valid %, then slot F1.",
        "",
        "| Dataset | Format | Exact match % | Format-valid % | Intent acc % | Slot F1 % | N |",
        "|---------|--------|---------------|----------------|--------------|-----------|---|",
    ]
    for ds, fmt, m in rows:
        lines.append(
            f"| {ds} | {fmt} | {m['exact_match']:.2f} | {m['format_valid_pct']:.2f} | "
            f"{m['intent_accuracy']:.2f} | {m['slot_f1']:.2f} | {m['num_examples']} |"
        )
    lines += [
        "",
        "## Means across datasets",
        "",
        "| Format | Mean exact match % | Mean format-valid % | Mean slot F1 % |",
        "|--------|--------------------|---------------------|----------------|",
    ]
    for fmt in ("json", "key_value"):
        mm = means[fmt]
        lines.append(
            f"| {fmt} | {mm['exact_match']:.2f} | {mm['format_valid_pct']:.2f} | {mm['slot_f1']:.2f} |"
        )
    lines += [
        "",
        f"## Selected format: `{selected}`",
        "",
        f"Chosen by mean exact-match "
        f"(json={means['json']['exact_match']:.2f}, key_value={means['key_value']['exact_match']:.2f})",
        f"with tie-breakers format-valid % "
        f"(json={means['json']['format_valid_pct']:.2f}, key_value={means['key_value']['format_valid_pct']:.2f}) "
        f"and slot F1 "
        f"(json={means['json']['slot_f1']:.2f}, key_value={means['key_value']['slot_f1']:.2f}).",
        "",
    ]
    out = RES / "format_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text())
    print(f"SELECTED_FORMAT={selected}")
    (RES / "selected_format.txt").write_text(selected + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
