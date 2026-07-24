#!/usr/bin/env python3
"""Run frozen Qwen2.5-0.5B-Instruct on SNIPS / ATIS (no training)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formats import parse_prediction  # noqa: E402
from evaluate import evaluate_predictions, load_jsonl  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_labels(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(template: str, request: str, intent_labels: str, slot_labels: str) -> str:
    return (
        template.replace("{intent_labels}", intent_labels)
        .replace("{slot_labels}", slot_labels)
        .replace("{request}", request)
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    device: str,
) -> List[str]:
    texts = [to_chat_text(tokenizer, p) for p in prompts]
    # Left-padding is required for correct batched causal LM generation.
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    # With left-padding, every row shares the same padded input length; new
    # tokens always start after that shared width (not after each unpadded length).
    padded_input_len = encoded["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.9
    else:
        # Disable sampling-related defaults that some model configs still set.
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0

    with torch.no_grad():
        out = model.generate(**encoded, **gen_kwargs)

    decoded: List[str] = []
    for seq in out:
        gen_ids = seq[padded_input_len:]
        decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return decoded


def measure_token_lengths(
    tokenizer,
    template: str,
    records: List[dict],
    intent_labels: str,
    slot_labels: str,
) -> Dict[str, Any]:
    lengths = []
    for rec in records:
        prompt = build_prompt(template, rec["request"], intent_labels, slot_labels)
        text = to_chat_text(tokenizer, prompt)
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        lengths.append(len(ids))
    return {
        "num_examples": len(lengths),
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "p95_tokens": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
    }


def run_inference(
    dataset: str,
    split: str,
    fmt: str,
    config: dict,
    processed_dir: Path,
    prompts_dir: Path,
    results_dir: Path,
    limit: Optional[int] = None,
    check_lengths_only: bool = False,
) -> Path:
    model_name = config["model_name"]
    max_input_length = int(config["max_input_length"])
    max_new_tokens = int(config["max_new_tokens"])
    do_sample = bool(config["do_sample"])
    seed = int(config["seed"])
    batch_size = int(config.get("batch_size", 1))

    set_seed(seed)
    device = pick_device()
    print(f"Device: {device}")

    gold_path = processed_dir / f"{dataset}_{split}.jsonl"
    records = load_jsonl(gold_path)
    if limit is not None:
        records = records[:limit]

    intent_labels = read_labels(processed_dir / f"{dataset}_intent_labels.txt")
    slot_labels = read_labels(processed_dir / f"{dataset}_slot_labels.txt")
    template = (prompts_dir / f"{fmt}.txt").read_text(encoding="utf-8")

    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    length_stats = measure_token_lengths(
        tokenizer, template, records, intent_labels, slot_labels
    )
    print(f"Token length stats ({dataset}/{split}/{fmt}): {length_stats}")
    if length_stats["max_tokens"] > max_input_length:
        new_len = int(length_stats["max_tokens"] + 8)
        print(
            f"WARNING: max prompt tokens {length_stats['max_tokens']} > "
            f"max_input_length {max_input_length}. Raising to {new_len}."
        )
        max_input_length = new_len
        config["max_input_length"] = new_len

    if check_lengths_only:
        out = results_dir / f"{dataset}_{split}_{fmt}_token_stats.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(length_stats, indent=2) + "\n", encoding="utf-8")
        return out

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Hard guarantee: no training artifacts
    for p in model.parameters():
        p.requires_grad = False

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

    predictions: List[dict] = []
    for start in tqdm(range(0, len(records), batch_size), desc=f"{dataset}/{split}/{fmt}"):
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
            parsed = parse_prediction(raw, fmt)
            predictions.append(
                {
                    "id": rec["id"],
                    "request": rec["request"],
                    "gold": {"intent": rec["intent"], "slots": rec["slots"]},
                    "raw_prediction": raw,
                    "parsed_prediction": parsed,
                    "format": fmt,
                    "metadata": metadata,
                }
            )

    if split == "test":
        pred_path = results_dir / f"{dataset}_test_predictions.jsonl"
        metrics_path = results_dir / f"{dataset}_test_metrics.json"
    else:
        pred_path = results_dir / f"{dataset}_{split}_{fmt}_predictions.jsonl"
        metrics_path = results_dir / f"{dataset}_{split}_{fmt}_metrics.json"

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = evaluate_predictions(records, predictions, fmt)
    metrics["metadata"] = metadata
    metrics["token_length_stats"] = length_stats
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Qwen base-model runner")
    parser.add_argument("--dataset", choices=("snips", "atis"), required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--format", choices=("json", "key_value"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "base_qwen.yaml",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=ROOT / "prompts",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "base",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit")
    parser.add_argument(
        "--check-lengths-only",
        action="store_true",
        help="Only measure prompt token lengths (still loads tokenizer)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model_name"] = args.model
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    run_inference(
        dataset=args.dataset,
        split=args.split,
        fmt=args.format,
        config=config,
        processed_dir=args.processed_dir,
        prompts_dir=args.prompts_dir,
        results_dir=args.results_dir,
        limit=args.limit,
        check_lengths_only=args.check_lengths_only,
    )


if __name__ == "__main__":
    main()
