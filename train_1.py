"""
Training script for the from-scratch decoder-only Transformer.

What this does:
1. Loads WikiText-103 (falls back to a local .txt file if datasets/HF is unavailable)
2. Tokenizes with GPT-2 BPE (tiktoken)
3. Trains with AdamW + linear warmup -> cosine decay
4. Logs loss every N steps and saves a loss curve plot
5. Generates a few sample completions at the end

Run baseline (learned absolute positional embeddings):
  python train.py

Run RoPE model (rotary positional embeddings in attention):
  python train.py --use_rope

Run both back-to-back and print a comparison table:
  python train.py --compare
"""

import argparse
import math
import time
import torch
import tiktoken
import matplotlib.pyplot as plt

from model_1 import DecoderOnlyTransformer

# ---------------------------------------------------------------------------
# Config — ~51M params with tied embeddings (GPT-2 vocab)
# ---------------------------------------------------------------------------
CONTEXT_LEN = 256
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
FF_DIM = 2048

MAX_STEPS = 500
EVAL_INTERVAL = 25
EVAL_ITERS = 3
WARMUP_STEPS = 50
LR = 3e-4
SEED = 42

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# MPS memory is tighter than CUDA; keep a smaller batch there.
BATCH_SIZE = 8 if DEVICE == "cuda" else 4

if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
elif DEVICE in ("cuda", "mps"):
    DTYPE = torch.float16
else:
    DTYPE = torch.float32


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Cap tokens so local MPS/CPU runs finish in a reasonable time while still
# using real WikiText-103 text (full corpus is ~100M+ tokens).
MAX_TOKENS = 2_000_000


def load_wikitext_tokens(tokenizer):
    """Load WikiText-103 and tokenize up to MAX_TOKENS."""
    try:
        from datasets import load_dataset

        print("Loading WikiText-103 from HuggingFace...")
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
        )
        print("Tokenizing (streaming until token budget)...")
        tokens = []
        for text in ds["text"]:
            if not text or not text.strip():
                continue
            tokens.extend(tokenizer.encode(text + "\n"))
            if len(tokens) >= MAX_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                break
        return tokens
    except Exception as e:
        print(f"Could not load WikiText-103 ({e}). Falling back to local train.txt")
        with open("train.txt", "r", encoding="utf-8") as f:
            return tokenizer.encode(f.read())


def prepare_data(tokenizer):
    tokens = load_wikitext_tokens(tokenizer)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"Total tokens: {len(data):,}")

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    return train_data, val_data


def get_batch(split, train_data, val_data):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([d[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([d[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LR * (1 + math.cos(math.pi * progress))


def train_one(use_rope, train_data, val_data, tokenizer):
    set_seed(SEED)
    label = "rope" if use_rope else "baseline"
    print("\n" + "=" * 72)
    print(f"Training {'RoPE' if use_rope else 'BASELINE (absolute positional embeddings)'}")
    print("=" * 72)

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        context_len=CONTEXT_LEN,
        use_rope=use_rope,
    ).to(DEVICE)

    total_params = model.num_params()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"use_rope={use_rope}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    val_steps = []
    initial_train_loss = None
    initial_val_loss = None

    start_time = time.time()
    use_autocast = DEVICE in ("cuda", "mps") and DTYPE in (torch.float16, torch.bfloat16)

    for step in range(MAX_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch("train", train_data, val_data)

        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=use_autocast):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())
        elapsed = time.time() - start_time

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data)
            val_losses.append(losses["val"])
            val_steps.append(step)
            if initial_train_loss is None:
                initial_train_loss = losses["train"]
                initial_val_loss = losses["val"]
            print(
                f"[{label}] step {step:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | elapsed {elapsed:.1f}s"
            )
        else:
            print(f"[{label}] step {step:5d} | train loss {loss.item():.4f} | elapsed {elapsed:.1f}s")

    # Loss curve
    curve_path = f"loss_curve_{label}.png"
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train loss (per step)", alpha=0.5)
    plt.plot(val_steps, val_losses, label="val loss", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    title = "RoPE" if use_rope else "Baseline (absolute PE)"
    plt.title(f"WikiText Pretraining Loss — {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved {curve_path}")

    final_losses = estimate_loss(model, train_data, val_data)
    val_perplexity = math.exp(min(final_losses["val"], 20))
    print(f"\n[{label}] Initial train loss: {initial_train_loss:.4f}")
    print(f"[{label}] Initial val loss:   {initial_val_loss:.4f}")
    print(f"[{label}] Final train loss:   {final_losses['train']:.4f}")
    print(f"[{label}] Final val loss:     {final_losses['val']:.4f}")
    print(f"[{label}] Validation perplexity: {val_perplexity:.2f}")

    print(f"\n[{label}] --- Sample generations ---")
    prompts = [
        "The history of",
        "In mathematics,",
        "According to the",
    ]
    for p in prompts:
        idx = torch.tensor([tokenizer.encode(p)], dtype=torch.long).to(DEVICE)
        out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=40)
        print(f"\nPrompt: {p!r}")
        print("Output:", tokenizer.decode(out[0].tolist()))

    ckpt_path = f"stage1_model_{label}.pt"
    torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": MAX_STEPS,
        "use_rope": use_rope,
    },
    ckpt_path,
)
    print(f"\nSaved model weights to {ckpt_path}")

    return {
        "label": "RoPE" if use_rope else "Baseline",
        "params_m": total_params / 1e6,
        "initial_train": initial_train_loss,
        "initial_val": initial_val_loss,
        "final_train": final_losses["train"],
        "final_val": final_losses["val"],
        "perplexity": val_perplexity,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_steps": val_steps,
    }


def print_comparison(results):
    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    header = (
        f"{'Model':<12} {'Init val':>10} {'Final val':>10} "
        f"{'Final ppl':>10} {'Params':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<12} {r['initial_val']:10.4f} {r['final_val']:10.4f} "
            f"{r['perplexity']:10.2f} {r['params_m']:9.2f}M"
        )

    # Combined loss curve
    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(r["val_steps"], r["val_losses"], marker="o", label=f"{r['label']} val")
    plt.xlabel("Step")
    plt.ylabel("Validation loss")
    plt.title("WikiText: Baseline vs RoPE Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve_comparison.png",dpi=300,bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("\nSaved loss_curve_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_rope",
        action="store_true",
        help="Use rotary positional embeddings instead of absolute PE",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train baseline then RoPE and print a comparison table",
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}, dtype: {DTYPE}")
    set_seed(SEED)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_data, val_data = prepare_data(tokenizer)

    if args.compare:
        results = [
            train_one(False, train_data, val_data, tokenizer),
            train_one(True, train_data, val_data, tokenizer),
        ]
        print_comparison(results)
    else:
        train_one(args.use_rope, train_data, val_data, tokenizer)


if __name__ == "__main__":
    main()
