#!/usr/bin/env python3
"""
Stage 2 — Direct Preference Optimization (DPO)

Trains on top of the SFT checkpoint using preference pairs.

Input:
    data/processed/{snips,atis}_{train,validation}_pairs_dpo.jsonl

Output:
    checkpoints/dpo/final
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from trl import DPOConfig, DPOTrainer

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

SFT_CHECKPOINT = Path("checkpoints/sft/final")
CHECKPOINT_DIR = Path("checkpoints/dpo")

MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 512

BATCH_SIZE = 2
NUM_EPOCHS = 1

LEARNING_RATE = 5e-6
BETA = 0.1
WARMUP_RATIO = 0.10

SEED = 455

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
)
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    choices=["snips", "atis", "both"],
    default="snips",
)

parser.add_argument(
    "--data_dir",
    default="data/processed",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=NUM_EPOCHS,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
)

args = parser.parse_args()

data_dir = Path(args.data_dir)
datasets = (
    ["snips", "atis"]
    if args.dataset == "both"
    else [args.dataset]
)

train_files = [
    str(data_dir / f"{d}_train_pairs_dpo.jsonl")
    for d in datasets
]

validation_files = [
    str(data_dir / f"{d}_validation_pairs_dpo.jsonl")
    for d in datasets
]

train_dataset = load_dataset(
    "json",
    data_files=train_files,
    split="train",
)

validation_dataset = load_dataset(
    "json",
    data_files=validation_files,
    split="train",
)

print(f"Training examples   : {len(train_dataset)}")
print(f"Validation examples : {len(validation_dataset)}")
print(f"Loading SFT checkpoint from {SFT_CHECKPOINT}")

tokenizer = AutoTokenizer.from_pretrained(
    SFT_CHECKPOINT
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    SFT_CHECKPOINT
)


print("Loading reference model...")

ref_model = AutoModelForCausalLM.from_pretrained(
    SFT_CHECKPOINT
)


ref_model.eval()

for param in ref_model.parameters():
    param.requires_grad = False

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

training_args = DPOConfig(
    output_dir=str(CHECKPOINT_DIR),
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=LEARNING_RATE,
    num_train_epochs=args.epochs,
    max_steps=1000,
    beta=BETA,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    max_length=MAX_LENGTH,

    remove_unused_columns=False,
    report_to="none",
    seed=SEED,
)
print("Initializing DPO trainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
)
print("\nStarting DPO training...")
print(f"Device           : {DEVICE}")
print(f"Training samples : {len(train_dataset)}")
print(f"Validation       : {len(validation_dataset)}")
print(f"Epochs           : {args.epochs}")
print(f"Batch size       : {args.batch_size}")
print(f"Learning rate    : {LEARNING_RATE}")
print(f"Beta             : {BETA}")

trainer.train()
print("\nSaving DPO model...")


final_path = CHECKPOINT_DIR / "final"

trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))

print(f"DPO model saved to {final_path}")
print("\nTraining complete!")

print(
    "\nGenerate predictions with:\n"
    f"python src/generate_predictions.py "
    f"--checkpoint {final_path} "
    "--dataset snips "
    "--split validation "
    "--out results/dpo/snips_validation_predictions.jsonl"
)

print(
    "\nThen evaluate using:\n"
    "python src/evaluate.py "
    "--gold data/processed/snips_validation.jsonl "
    "--predictions results/dpo/snips_validation_predictions.jsonl "
    "--output results/dpo/snips_validation_metrics.json "
    "--format json"
)