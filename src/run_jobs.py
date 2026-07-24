#!/usr/bin/env python3
"""Load Qwen once and run multiple frozen base evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import evaluate_predictions, load_jsonl  # noqa: E402
from formats import parse_prediction  # noqa: E402
from run_base import (  # noqa: E402
    build_prompt,
    generate_batch,
    load_config,
    measure_token_lengths,
    pick_device,
    read_labels,
    set_seed,
)


def run_one(
    model,
    tokenizer,
    dataset: str,
    split: str,
    fmt: str,
    config: dict,
    processed_dir: Path,
    prompts_dir: Path,
    results_dir: Path,
    limit: Optional[int] = None,
) -> dict:
    max_input_length = int(config["max_input_length"])
    max_new_tokens = int(config["max_new_tokens"])
    do_sample = bool(config["do_sample"])
    seed = int(config["seed"])
    batch_size = int(config.get("batch_size", 1))
    model_name = config["model_name"]
    device = pick_device()

    gold_path = processed_dir / f"{dataset}_{split}.jsonl"
    gold_full = load_jsonl(gold_path)
    records = gold_full if limit is None else gold_full[:limit]
    is_partial = len(records) < len(gold_full)

    intent_labels = read_labels(processed_dir / f"{dataset}_intent_labels.txt")
    slot_labels = read_labels(processed_dir / f"{dataset}_slot_labels.txt")
    template = (prompts_dir / f"{fmt}.txt").read_text(encoding="utf-8")

    length_stats = measure_token_lengths(
        tokenizer, template, records, intent_labels, slot_labels
    )
    print(f"Token length stats ({dataset}/{split}/{fmt}): {length_stats}")
    if length_stats["max_tokens"] > max_input_length:
        max_input_length = int(length_stats["max_tokens"] + 8)
        print(f"Raising max_input_length to {max_input_length}")

    metadata = {
        "model_name": model_name,
        "format": fmt,
        "dataset": dataset,
        "split": split,
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "seed": seed,
        "batch_size": batch_size,
        "device": device,
    }

    predictions = []
    pred_path_tmp = results_dir / (
        f"{dataset}_test_predictions.jsonl.tmp"
        if split == "test"
        else f"{dataset}_{split}_{fmt}_predictions.jsonl.tmp"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    # Resume from a previous interrupted run if a temp file exists.
    start_idx = 0
    if pred_path_tmp.exists():
        with pred_path_tmp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "id" in obj:
                    predictions.append(obj)
        # Deduplicate by id, preserve first occurrence order.
        seen = set()
        deduped = []
        for row in predictions:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            deduped.append(row)
        predictions = deduped
        # Rewrite a clean checkpoint so resume indices match file length.
        with pred_path_tmp.open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        start_idx = len(predictions)
        if start_idx:
            print(f"Resuming from example {start_idx}/{len(records)}")

    mode = "a" if start_idx else "w"
    with pred_path_tmp.open(mode, encoding="utf-8") as out_f:
        for start in tqdm(
            range(start_idx, len(records), batch_size),
            desc=f"{dataset}/{split}/{fmt}",
            total=(len(records) - start_idx + batch_size - 1) // batch_size,
        ):
            batch = records[start : start + batch_size]
            prompts = [
                build_prompt(template, rec["request"], intent_labels, slot_labels)
                for rec in batch
            ]
            raws = generate_batch(
                model,
                tokenizer,
                prompts,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                device=device,
            )
            for rec, raw in zip(batch, raws):
                row = {
                    "id": rec["id"],
                    "request": rec["request"],
                    "gold": {"intent": rec["intent"], "slots": rec["slots"]},
                    "raw_prediction": raw,
                    "parsed_prediction": parse_prediction(raw, fmt),
                    "format": fmt,
                    "metadata": metadata,
                }
                predictions.append(row)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    if split == "test":
        pred_path = results_dir / f"{dataset}_test_predictions.jsonl"
        metrics_path = results_dir / f"{dataset}_test_metrics.json"
    else:
        pred_path = results_dir / f"{dataset}_{split}_{fmt}_predictions.jsonl"
        metrics_path = results_dir / f"{dataset}_{split}_{fmt}_metrics.json"

    if is_partial:
        print(
            f"Partial run ({len(predictions)}/{len(gold_full)}); "
            f"keeping checkpoint at {pred_path_tmp}"
        )
        return {"num_examples": len(predictions), "partial": True}

    # Write final predictions from memory (authoritative), then drop checkpoint.
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if pred_path_tmp.exists():
        pred_path_tmp.unlink()

    metrics = evaluate_predictions(records, predictions, fmt)
    metrics["metadata"] = metadata
    metrics["token_length_stats"] = length_stats
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in (
        "num_examples", "format_valid_pct", "intent_accuracy", "slot_f1", "exact_match"
    )}, indent=2))
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        nargs="+",
        required=True,
        help="Jobs as dataset:split:format (e.g. snips:validation:json)",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base_qwen.yaml")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "base")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading {config['model_name']} once for {len(args.jobs)} job(s)")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    for job in args.jobs:
        dataset, split, fmt = job.split(":")
        print(f"\n===== {dataset} / {split} / {fmt} =====")
        run_one(
            model,
            tokenizer,
            dataset=dataset,
            split=split,
            fmt=fmt,
            config=config,
            processed_dir=args.processed_dir,
            prompts_dir=args.prompts_dir,
            results_dir=args.results_dir,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
