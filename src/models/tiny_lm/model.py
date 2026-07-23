import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generation import sample_next_token


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input shape:
        x: [B, T, C]

    Output shape:
        out: [B, T, C]

    B = batch size
    T = sequence length / block size
    C = embedding dimension
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # One linear layer produces Q, K, V together
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)

        # Final projection after attention
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        # Shape: [1, 1, block_size, block_size]
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)

        # Reshape into multiple heads
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0,
                float("-inf"),
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Merge heads
        # [B, n_head, T, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    """
    Position-wise MLP used inside each Transformer block.
    """

    def __init__(self, n_embd, d_ff=None, dropout=0.1):
        super().__init__()

        hidden_size = int(d_ff) if d_ff is not None else 4 * n_embd
        if hidden_size < 1:
            raise ValueError("d_ff must be positive.")

        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One GPT-style Transformer decoder block.

    Uses pre-layer normalization:
        x = x + attention(layernorm(x))
        x = x + mlp(layernorm(x))
    """

    def __init__(self, n_embd, n_head, block_size, d_ff=None, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd, d_ff=d_ff, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Small GPT-style decoder-only language model.

    Input:
        idx: [B, T]

    Output:
        logits: [B, T, vocab_size]
        loss: scalar if targets are provided
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=2,
        n_head=4,
        n_embd=128,
        d_ff=None,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT-style small normal initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Input sequence length {T} exceeds block_size {self.block_size}"
            )

        token_emb = self.token_embedding(idx)  # [B, T, n_embd]

        positions = torch.arange(0, T, device=idx.device)
        pos_emb = self.position_embedding(positions)  # [T, n_embd]

        x = token_emb + pos_emb  # [B, T, n_embd]

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        generator=None,
    ):
        """Autoregressively sample new tokens from a character-level prompt."""

        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch_size, sequence_length].")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if idx.shape[1] < 1:
            raise ValueError("idx must contain at least one prompt token.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            next_id = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            idx = torch.cat([idx, next_id], dim=1)

        return idx
