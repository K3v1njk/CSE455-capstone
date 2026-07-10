"""
Training script for the from-scratch decoder-only Transformer.

What this does:
1. Loads TinyStories (falls back to a local .txt file if datasets/HF is unavailable)
2. Tokenizes with GPT-2 BPE (tiktoken)
3. Trains with AdamW + linear warmup -> cosine decay
4. Logs loss every N steps and saves a loss curve plot
5. Generates a few sample completions at the end

Run: python train.py
"""

import math
import time
import torch
import tiktoken
import matplotlib.pyplot as plt

from model import DecoderOnlyTransformer

# ---------------------------------------------------------------------------
# Config — matches the proposal's technical assumptions
# ---------------------------------------------------------------------------
CONTEXT_LEN = 256
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
FF_DIM = 1536

BATCH_SIZE = 8
MAX_STEPS = 150
EVAL_INTERVAL = 25
EVAL_ITERS = 3
WARMUP_STEPS = 200
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab


def load_text():
    """Load TinyStories via HuggingFace datasets. Falls back to a local file
    called train.txt if you don't have internet access / the datasets lib."""
    try:
        from datasets import load_dataset
        print("Loading TinyStories from HuggingFace...")
        ds = load_dataset("roneneldan/TinyStories", split="train[:5%]")  # subset for speed
        text = "\n".join(ds["text"])
        return text
    except Exception as e:
        print(f"Could not load TinyStories ({e}). Falling back to local train.txt")
        with open("train.txt", "r", encoding="utf-8") as f:
            return f.read()


print("Tokenizing...")
raw_text = load_text()
tokens = tokenizer.encode(raw_text)
data = torch.tensor(tokens, dtype=torch.long)
print(f"Total tokens: {len(data):,}")

# 90/10 train/val split
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]
print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + CONTEXT_LEN] for i in ix])
    y = torch.stack([d[i + 1:i + CONTEXT_LEN + 1] for i in ix])  # shifted by 1 = next-token targets
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Model + optimizer + LR schedule
# ---------------------------------------------------------------------------
model = DecoderOnlyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    context_len=CONTEXT_LEN,
).to(DEVICE)

print(f"Model parameters: {model.num_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def get_lr(step):
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LR * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
train_losses = []
val_losses = []
val_steps = []

start_time = time.time()

for step in range(MAX_STEPS):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    x, y = get_batch("train")

    with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
        logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    train_losses.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        val_losses.append(losses["val"])
        val_steps.append(step)
        elapsed = time.time() - start_time
        print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
              f"elapsed {elapsed:.1f}s")
    else:
        elapsed = time.time() - start_time
        print(f"step {step:5d} | train loss {loss.item():.4f} | elapsed {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Save loss curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train loss (per step)", alpha=0.5)
plt.plot(val_steps, val_losses, label="val loss", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Stage 1 Pretraining Loss Curve")
plt.legend()
plt.savefig("loss_curve.png", dpi=150)
print("Saved loss_curve.png")

# ---------------------------------------------------------------------------
# Validation perplexity
# ---------------------------------------------------------------------------
final_losses = estimate_loss(model)
val_perplexity = math.exp(final_losses["val"])
print(f"\nFinal train loss: {final_losses['train']:.4f}")
print(f"Final val loss: {final_losses['val']:.4f}")
print(f"Validation perplexity: {val_perplexity:.2f}")

# ---------------------------------------------------------------------------
# Sample generations
# ---------------------------------------------------------------------------
print("\n--- Sample generations ---")
prompts = ["Once upon a time", "The little dog", "One day, a girl"]
for p in prompts:
    idx = torch.tensor([tokenizer.encode(p)], dtype=torch.long).to(DEVICE)
    out = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=40)
    print(f"\nPrompt: {p!r}")
    print("Output:", tokenizer.decode(out[0].tolist()))

torch.save(model.state_dict(), "stage1_model.pt")
print("\nSaved model weights to stage1_model.pt")
