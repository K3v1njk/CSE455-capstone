"""
Decoder-only Transformer, built from scratch in PyTorch.
No nn.Transformer, no high-level Trainer/fit wrappers.

Supports two positional encoding modes:
- Baseline: learned absolute positional embeddings
- RoPE: rotary positional embeddings applied to Q/K in every attention head

Default size (~50M with tied embeddings, GPT-2 vocab):
- 8 transformer blocks
- 8 attention heads
- 512 embedding dimension
- 512 token context length
- Feed-forward dim = 2048 (4x embedding dim)
- Pre-layernorm, GELU, residual connections, causal masking
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for attention queries and keys."""

    def __init__(self, head_dim, context_len, base=10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")

        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(context_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)

        cos = frequencies.cos().repeat_interleave(2, dim=-1)
        sin = frequencies.sin().repeat_interleave(2, dim=-1)

        # Shape: (1, 1, context_len, head_dim)
        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(0), persistent=False)

    @staticmethod
    def rotate_pairs(x):
        """Rotate adjacent pairs: (x0, x1) -> (-x1, x0)."""
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        return rotated.flatten(-2)

    def forward(self, q, k):
        sequence_length = q.size(-2)
        cos = self.cos_cached[:, :, :sequence_length].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:, :, :sequence_length].to(device=q.device, dtype=q.dtype)
        q_rotated = (q * cos) + (self.rotate_pairs(q) * sin)
        k_rotated = (k * cos) + (self.rotate_pairs(k) * sin)
        return q_rotated, k_rotated

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (autoregressive) mask."""

    def __init__(self, embed_dim, num_heads, context_len, dropout=0.1, use_rope=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if use_rope:
            self.rotary_embedding = RotaryEmbedding(
                head_dim=self.head_dim,
                context_len=context_len,
            )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)).view(1, 1, context_len, context_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self.rotary_embedding(q, k)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:, :, :T, :T],float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """
    SwiGLU FeedForward Network.
    """

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.gate_proj = nn.Linear(embed_dim, ff_dim)
        self.value_proj = nn.Linear(embed_dim, ff_dim)
        self.out_proj = nn.Linear(ff_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x)

        x = gate * value

        x = self.out_proj(x)

        return self.dropout(x)


class DecoderBlock(nn.Module):
    """One transformer decoder block: pre-LN self-attention + pre-LN feedforward."""

    def __init__(self, embed_dim, num_heads, ff_dim, context_len, dropout=0.1, use_rope=False):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, context_len, dropout, use_rope)
        self.ln2 = RMSNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x):
        # Pre-layernorm + residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        ff_dim=2048,
        context_len=512,
        dropout=0.1,
        use_rope=False,
    ):
        super().__init__()
        self.context_len = context_len
        self.use_rope = use_rope

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        # Absolute positional embeddings only for the baseline (no RoPE).
        self.pos_emb = None if use_rope else nn.Embedding(context_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, context_len, dropout, use_rope)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: output projection shares token embedding weights
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if module is self.lm_head:
                std = 0.02 / math.sqrt(2 * len(self.blocks))

            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_len, f"sequence length {T} exceeds context length {self.context_len}"

        x = self.token_emb(idx)
        if self.pos_emb is not None:
            positions = torch.arange(T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(positions)

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
)
        return logits, loss

    @torch.no_grad()

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate tokens, one at a time."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits, _ = self(idx_cond)
            temperature = max(temperature, 1e-5)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                assert top_k > 0, "top_k must be greater than 0"

                top_k = min(top_k, logits.size(-1))

                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        self.train()
        return idx

    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick sanity check: forward + backward pass on dummy data
    torch.manual_seed(0)
    vocab_size = 50257  # GPT-2 tokenizer vocab size

    for use_rope in (False, True):
        label = "RoPE" if use_rope else "baseline"
        model = DecoderOnlyTransformer(vocab_size=vocab_size, use_rope=use_rope)
        print(f"[{label}] Total parameters: {model.num_params():,} ({model.num_params() / 1e6:.2f}M)")

        dummy_input = torch.randint(0, vocab_size, (2, 64))
        dummy_targets = torch.randint(0, vocab_size, (2, 64))
        logits, loss = model(dummy_input, dummy_targets)
        print(f"[{label}] Logits shape: {logits.shape} | Loss: {loss.item():.4f}")
        loss.backward()
        print(f"[{label}] Backward pass succeeded.")
