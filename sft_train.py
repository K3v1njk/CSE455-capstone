"""
Stage 2 — Supervised Fine-Tuning (SFT) of Qwen2.5-0.5B-Instruct
on SNIPS/ATIS intent+slot structured-output data.

Written from scratch in PyTorch (forward pass, cross-entropy loss,
backprop, AdamW) per project requirements — no Trainer/fit() wrapper.

Input data schema (from teammate 1's base-model branch):
    data/processed/{snips,atis}_{train,validation}_pairs_json_sft.jsonl
    Each line: {"dataset": "...", "input": "<request text>", "target": "<JSON string>"}

Usage:
    python sft_train.py --dataset snips
    python sft_train.py --dataset atis
    python sft_train.py --dataset both   # trains on snips+atis combined
"""

import argparse
import json
import math
import time
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 768          # matches configs/base_qwen.yaml from the base-model branch
BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 1e-5
WARMUP_STEPS = 2
EVAL_INTERVAL = 100       # steps between validation checks
GRAD_CLIP = 1.0
CHECKPOINT_DIR = Path("checkpoints/sft")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Exact prompt from teammate 1's prompts/json.txt, so SFT trains on the same
# prompt distribution the base-model eval used — keeps all 3 checkpoints comparable.
PROMPT_TEMPLATE = (
    "You are a semantic parser.\n\n"
    "Convert the user request into exactly one valid JSON object using this schema:\n\n"
    "{{\"intent\":\"<allowed intent>\",\"slots\":[{{\"name\":\"<allowed slot name>\",\"value\":\"<text copied from the request>\"}}]}}\n\n"
    "Rules:\n"
    "- Use only the allowed intent and slot labels listed below.\n"
    "- If no slots are present, return an empty list for \"slots\".\n"
    "- Copy slot values from the request; do not invent values.\n"
    "- Return only the JSON object. Do not include an explanation, Markdown, or code fences.\n\n"
    "Allowed intents:\n{intent_labels}\n\n"
    "Allowed slots:\n{slot_labels}\n\n"
    "User request:\n{request}"
)


def load_labels(path):
    """Load a newline-separated label list file into a single comma-joined string
    for injection into the prompt."""
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return ", ".join(labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    """Loads (input, target) JSONL pairs and formats them as prompt + target
    sequences, masking the loss on the prompt tokens.

    Each example's prompt includes the allowed intent/slot labels for its
    dataset (SNIPS or ATIS), loaded from data/processed/{dataset}_intent_labels.txt
    and data/processed/{dataset}_slot_labels.txt, matching prompts/json.txt exactly.
    """

    def __init__(self, jsonl_paths, tokenizer, data_dir, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Cache label strings per dataset so we don't re-read the label files per example
        self._label_cache = {}

        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    dataset = row["dataset"]
                    if dataset not in self._label_cache:
                        intent_labels = load_labels(Path(data_dir) / f"{dataset}_intent_labels.txt")
                        slot_labels = load_labels(Path(data_dir) / f"{dataset}_slot_labels.txt")
                        self._label_cache[dataset] = (intent_labels, slot_labels)
                    self.examples.append((row["input"], row["target"], dataset))
        print(f"Loaded {len(self.examples)} examples from {jsonl_paths}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        request, target, dataset = self.examples[idx]
        intent_labels, slot_labels = self._label_cache[dataset]
        prompt = PROMPT_TEMPLATE.format(
            intent_labels=intent_labels, slot_labels=slot_labels, request=request
        )

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(" " + target, add_special_tokens=False)["input_ids"]
        eos_id = self.tokenizer.eos_token_id

        input_ids = prompt_ids + target_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + target_ids + [eos_id]

        # Truncate from the left on the prompt side if too long (keep target intact)
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    """Right-pad input_ids and labels to the same length within a batch."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        L = len(item["input_ids"])
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L] = item["labels"]
        attention_mask[i, :L] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def get_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=20):
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n += 1
    model.train()
    return total_loss / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["snips", "atis", "both"], default="snips")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    datasets = ["snips", "atis"] if args.dataset == "both" else [args.dataset]
    train_paths = [data_dir / f"{d}_train_pairs_json_sft.jsonl" for d in datasets]
    val_paths = [data_dir / f"{d}_validation_pairs_json_sft.jsonl" for d in datasets]

    print(f"Device: {DEVICE}")
    print(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train()

    train_ds = SFTDataset(train_paths, tokenizer, data_dir)
    val_ds = SFTDataset(val_paths, tokenizer, data_dir)
   
    random.seed(455)
    if len(train_ds.examples) > 20:
        train_ds.examples = random.sample(train_ds.examples, 20)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Total training steps: {total_steps}")
    step = 0
    start = time.time()

    for epoch in range(args.epochs):
        for batch in train_loader:
            lr = get_lr(step, total_steps, args.lr, WARMUP_STEPS)
            for g in optimizer.param_groups:
                g["lr"] = lr

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # ---- Forward pass ----
            outputs = model(**batch)
            loss = outputs.loss

            # ---- Backward pass ----
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if step % 1 == 0:
                elapsed = time.time() - start
                print(f"epoch {epoch} step {step:5d}/{total_steps} | "
                      f"train loss {loss.item():.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s")

            if step % EVAL_INTERVAL == 0 and step > 0:
                val_loss = evaluate(model, val_loader, DEVICE)
                print(f"  --> validation loss: {val_loss:.4f}")

            step += 1

        # Save a checkpoint at the end of each epoch
        ckpt_path = CHECKPOINT_DIR / f"epoch{epoch}"
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    final_val_loss = evaluate(model, val_loader, DEVICE)
    print(f"\nFinal validation loss: {final_val_loss:.4f}")

    final_path = CHECKPOINT_DIR / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final SFT model to {final_path}")
    print("Next step: run src/evaluate.py on this checkpoint to get intent accuracy + slot F1, "
          "then hand the checkpoint to the DPO teammate.")


if __name__ == "__main__":
    main()