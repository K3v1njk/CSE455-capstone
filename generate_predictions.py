"""
Generates predictions from a trained SFT checkpoint for scoring with
teammate 1's src/evaluate.py.

evaluate.py expects: --gold GOLD --predictions PREDICTIONS --output OUTPUT --format {json,key_value}

Gold file (from teammate 1): data/processed/{dataset}_validation.jsonl
    fields: id, dataset, request, intent, slots

This script reads the gold file's requests, runs them through the SFT
checkpoint, and writes a predictions.jsonl file with the same "id" field
so evaluate.py can match predictions to gold examples.

Usage:
    python generate_predictions.py --checkpoint checkpoints/sft/final \
        --dataset snips --split validation --out results/sft/snips_validation_predictions.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sft_train import PROMPT_TEMPLATE, load_labels

MAX_NEW_TOKENS = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["snips", "atis"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run on the first N examples (for quick testing before a full run)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    data_dir = Path(args.data_dir)
    intent_labels = load_labels(data_dir / f"{args.dataset}_intent_labels.txt")
    slot_labels = load_labels(data_dir / f"{args.dataset}_slot_labels.txt")

    gold_path = data_dir / f"{args.dataset}_{args.split}.jsonl"
    print(f"Reading gold examples from: {gold_path}")

    rows = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} examples")

    if args.limit is not None:
        rows = rows[:args.limit]
        print(f"Limiting to first {len(rows)} examples for this run")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, row in enumerate(rows):
            prompt = PROMPT_TEMPLATE.format(
                intent_labels=intent_labels, slot_labels=slot_labels, request=row["request"]
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            prediction_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            f_out.write(json.dumps({"id": row["id"], "prediction": prediction_text}) + "\n")

            if i % 10 == 0:
                print(f"  {i}/{len(rows)} predictions generated")

    print(f"Saved predictions to {out_path}")
    print("Next: python src/evaluate.py --gold <gold_path> --predictions <this file> "
          "--output <metrics.json> --format json")


if __name__ == "__main__":
    main()