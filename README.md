# Stage 1 — Decoder-Only Transformer (from scratch)

## Setup
```bash
pip install -r requirements.txt
```

## Files
- `model.py` — the Transformer architecture (decoder-only, multi-head self-attention, built from scratch — no `nn.Transformer`)
- `train.py` — pretraining script: loads TinyStories, tokenizes with GPT-2 BPE, trains, saves a loss curve and sample generations

## Run
```bash
python train.py
```

This will:
1. Download a subset of TinyStories (falls back to a local `train.txt` if you have no internet access)
2. Train for 3000 steps
3. Save `loss_curve.png`
4. Print validation perplexity and sample text generations
5. Save model weights to `stage1_model.pt`

## Model config
- 6 transformer blocks, 6 attention heads, 384 embedding dim, 256 context length
- ~20-30M parameters
- AdamW optimizer, linear warmup + cosine decay, mixed precision on GPU

## Sanity check
Run `python model.py` directly to test the architecture in isolation (forward + backward pass on dummy data, confirms shapes are correct).
