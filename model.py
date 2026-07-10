"""
Decoder-only Transformer, built from scratch in PyTorch.
No nn.Transformer, no high-level Trainer/fit wrappers.

Matches proposal spec:
- 6 transformer blocks
- 6 attention heads
- 384 embedding dimension
- 256 token context length
- Feed-forward dim = 1536 (4x embedding dim)
- ~20M parameters
- Pre-layernorm, GELU, residual connections, causal masking
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (autoregressive) mask."""

    def __init__(self, embed_dim, num_heads, context_len, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape  

        qkv = self.qkv_proj(x)  
        q, k, v = qkv.split(C, dim=2) 
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ v  
        out = out.transpose(1, 2).contiguous().view(B, T, C)  
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feedforward network with GELU activation."""

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """One transformer decoder block: pre-LN self-attention + pre-LN feedforward."""

    def __init__(self, embed_dim, num_heads, ff_dim, context_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
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
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        ff_dim=1536,
        context_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, context_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_len, f"sequence length {T} exceeds context length {self.context_len}"
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  
        x = self.token_emb(idx) + self.pos_emb(positions)  
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
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
            logits = logits[:, -1, :] / temperature  

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        self.train()
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick sanity check: forward + backward pass on dummy data
    torch.manual_seed(0)
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    model = DecoderOnlyTransformer(vocab_size=vocab_size)
    print(f"Total parameters: {model.num_params():,}")

    dummy_input = torch.randint(0, vocab_size, (4, 256))   # (batch=4, seq_len=256)
    dummy_targets = torch.randint(0, vocab_size, (4, 256))

    logits, loss = model(dummy_input, dummy_targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass succeeded — gradients computed.")
