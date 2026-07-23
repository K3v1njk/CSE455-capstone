#!/usr/bin/env python3
"""Evaluate structured-output predictions against gold JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formats import exact_match, parse_prediction, slot_f1_counts  # noqa: E402


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_from_record(rec: dict) -> dict:
    if "gold" in rec:
        return rec["gold"]
    return {"intent": rec["intent"], "slots": rec["slots"]}


def evaluate_predictions(
    gold_rows: List[dict],
    pred_rows: List[dict],
    fmt: str,
) -> Dict[str, Any]:
    gold_by_id = {r["id"]: gold_from_record(r) for r in gold_rows}
    if not pred_rows:
        raise ValueError("No predictions provided")

    # Prefer prediction ids; fall back to zip order if missing
    n = 0
    format_valid = 0
    intent_correct = 0
    exact = 0
    tp = fp_fn_pred = fp_fn_gold = 0

    missing = 0
    for pred in pred_rows:
        pid = pred.get("id")
        if pid is None or pid not in gold_by_id:
            missing += 1
            continue
        gold = gold_by_id[pid]
        n += 1

        parsed = pred.get("parsed_prediction")
        if parsed is None and "raw_prediction" in pred:
            parsed = parse_prediction(pred["raw_prediction"], fmt)
        elif parsed is None and "prediction" in pred:
            parsed = parse_prediction(pred["prediction"], fmt)

        if isinstance(parsed, dict) and "intent" in parsed and "slots" in parsed:
            format_valid += 1
        else:
            parsed = None

        if parsed is not None and parsed["intent"] == gold["intent"]:
            intent_correct += 1

        if exact_match(gold, parsed):
            exact += 1

        pred_slots = parsed["slots"] if parsed is not None else []
        t, pc, gc = slot_f1_counts(gold["slots"], pred_slots)
        tp += t
        fp_fn_pred += pc
        fp_fn_gold += gc

    precision = tp / fp_fn_pred if fp_fn_pred else 0.0
    recall = tp / fp_fn_gold if fp_fn_gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "num_examples": n,
        "missing_predictions": missing,
        "format_valid_pct": (format_valid / n * 100.0) if n else 0.0,
        "intent_accuracy": (intent_correct / n * 100.0) if n else 0.0,
        "slot_precision": precision * 100.0,
        "slot_recall": recall * 100.0,
        "slot_f1": f1 * 100.0,
        "exact_match": (exact / n * 100.0) if n else 0.0,
        "format_valid_count": format_valid,
        "intent_correct_count": intent_correct,
        "exact_match_count": exact,
        "slot_tp": tp,
        "slot_predicted": fp_fn_pred,
        "slot_gold": fp_fn_gold,
        "format": fmt,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base-model predictions")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("json", "key_value"),
        default=None,
        help="Override format; otherwise read from prediction metadata",
    )
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.predictions)

    fmt: Optional[str] = args.format
    if fmt is None:
        for row in pred_rows:
            if "format" in row:
                fmt = row["format"]
                break
            meta = row.get("metadata") or {}
            if "format" in meta:
                fmt = meta["format"]
                break
    if fmt is None:
        raise SystemExit(
            "Could not infer format; pass --format json|key_value"
        )

    metrics = evaluate_predictions(gold_rows, pred_rows, fmt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
