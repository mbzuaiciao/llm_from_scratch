import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
    """1.3 Single-head attention (explicit shapes)."""
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
        """
        Initializes a single-head self-attention module.
        - d_model: The input embedding dimension.
        - d_k: The dimension of the key, query, and value vectors.
        - dropout: The dropout rate for the attention weights.
        - trace_shapes: If True, prints tensor shapes at each step.
        """
        super().__init__()
        # Linear layers to project the input into query, key, and value vectors.
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass for single-head self-attention.
        - x: Input tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        # 1. Project input into query, key, and value.
        q = self.q(x)  # (B,T,d_k)
        k = self.k(x)  # (B,T,d_k)
        v = self.v(x)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"q {q.shape}  k {k.shape}  v {v.shape}")
        
        # 2. Calculate attention scores (scaled dot-product).
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,T,T)
        
        # 3. Apply causal mask to prevent attending to future tokens.
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        
        # 4. Apply softmax to get attention weights and apply dropout.
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        
        # 5. Compute the weighted sum of value vectors.
        out = torch.matmul(w, v)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"weights {w.shape}  out {out.shape}")
        return out, w