from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Core building blocks for a tiny GPT-style transformer.
# ---- Blocks (self-contained for isolation) ----
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        """
        A causal self-attention module.
        - n_embd: The total embedding dimension.
        - n_head: The number of attention heads.
        - dropout: The dropout rate.
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head
        # A single linear layer projects the input to Q, K, and V for all heads at once.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # A final linear layer to project the concatenated head outputs back to the embedding dimension.
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):  # (B,T,C)
        B, T, C = x.shape
        # Project inputs once, then reshape to pull out Q/K/V for each head.
        # qkv shape: (B, T, 3, n_head, d_head)
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        # Unbind along dim 2 to get separate Q, K, V tensors.
        q, k, v = qkv.unbind(dim=2)
        # Transpose to get shape (B, n_head, T, d_head) for attention calculation.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Use PyTorch's built-in scaled dot-product attention.
        # It's efficient and handles causal masking (`is_causal=True`) and dropout internally.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        # Concatenate heads and project back to the original embedding dimension.
        # y shape: (B, n_head, T, d_head) -> (B, T, n_head, d_head) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # A standard two-layer MLP used in transformer blocks.
        # It expands the embedding dimension, applies a non-linearity, and then projects back.
        self.net = nn.Sequential(
            # Expansion layer
            nn.Linear(n_embd, mult * n_embd),
            # Non-linearity
            nn.GELU(),
            # Projection layer
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        # Layer normalization before the attention module.
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        # Layer normalization before the feed-forward network.
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(self, x):
        # This is a "pre-norm" architecture, where normalization is applied before the main operation.
        # Residual connection around the attention block.
        x = x + self.attn(self.ln1(x))
        # Residual connection around the feed-forward block.
        x = x + self.ffn(self.ln2(x))
        return x

# ---- Tiny GPT ----
class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        # Embedding layer for token indices.
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # Embedding layer for token positions.
        self.pos_emb = nn.Embedding(block_size, n_embd)
        # Dropout layer applied after embeddings.
        self.drop = nn.Dropout(dropout)
        # A stack of `n_layer` identical transformer blocks.
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        # Final layer normalization.
        self.ln_f = nn.LayerNorm(n_embd)
        # The final linear layer that maps from the embedding dimension to the vocabulary size.
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Apply a custom weight initialization.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initializes weights for Linear and Embedding layers, a common practice for GPT-style models
        to aid in stable training.
        """
        if isinstance(m, nn.Linear):
            # Normal initialization for linear layer weights.
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        # Create position indices from 0 to T-1.
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        # Get token and position embeddings and sum them up.
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        # Pass the input through the stack of transformer blocks.
        for blk in self.blocks:
            x = blk(x)
        # Apply the final layer normalization.
        x = self.ln_f(x)
        # Get the final logits from the head.
        logits = self.head(x)

        # If targets are provided, calculate the cross-entropy loss.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                top_k: int | None = 50, top_p: float | None = None):
        """
        Generates a sequence of tokens autoregressively.
        """
        from utils import top_k_top_p_filtering
        self.eval()
        # Guard: if the prompt is empty, start with a newline byte (10)
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            # Crop the context to the maximum block size.
            idx_cond = idx[:, -self.block_size:]
            # Forward pass to get logits for the current context.
            logits, _ = self(idx_cond)
            # Focus on the logits for the very last token.
            # Apply temperature scaling to control the randomness of predictions.
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            # Apply top-k and/or top-p (nucleus) filtering to the logits.
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            # Convert logits to probabilities.
            probs = torch.softmax(logits, dim=-1)
            # Sample the next token from the probability distribution.
            next_id = torch.multinomial(probs, num_samples=1)
            # Append the sampled token to the sequence.
            idx = torch.cat([idx, next_id], dim=1)
        return idx
