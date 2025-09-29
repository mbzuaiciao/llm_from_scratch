from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache  # your existing class

class CausalSelfAttentionModern(nn.Module):
    """A modern causal self-attention module incorporating RoPE, GQA, and sliding window attention."""
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        # For Grouped-Query Attention (GQA), n_kv_head is the number of K/V heads.
        # If not specified, it defaults to n_head (Multi-Head Attention).
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        # The number of Q heads that share a single K/V head.
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        # Linear projections for Query, Key, and Value.
        # Under GQA, Q has n_head heads, while K and V have n_kv_head heads.
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        # Final output projection.
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RoPE parameters and cache.
        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        # Sliding window attention parameters.
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

    def _maybe_init_rope(self, device):
        """Initializes the RoPE cache if it hasn't been created yet."""
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """
        Forward pass for modern causal self-attention.
        - x: Input tensor of shape (B, T, C).
        - kv_cache: Optional cache for keys and values for faster generation.
        - start_pos: The starting position for RoPE, used during generation.
        """
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # 1. Project inputs to Q, K, V and reshape for multi-head attention.
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)

        # 2. Apply RoPE to the current query and key tokens.
        # Cached keys/values are not rotated again.
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)   # (B,H, T,D)
            k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)

        # 3. If a KV cache is provided, concatenate the cached K/V with the current K/V.
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # 4. Apply sliding window attention by cropping the key/value cache.
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            # Keep the first 's' sink tokens and the last 'sliding_window' tokens.
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # 5. For GQA, repeat the K/V heads to match the number of Q heads.
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
        else:
            k_attn, v_attn = k_all, v_all

        # 6. Perform scaled dot-product attention.
        # `is_causal=True` is used during training/prefill to apply a causal mask.
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)          # (B,H,T,D)
        # 7. Reshape and project the output back to the embedding dimension.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # 8. Update the KV cache with the new keys and values.
        # We store the compact K/V (Hk heads), not the expanded k_attn/v_attn.
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk,*,D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        return y, new_cache
