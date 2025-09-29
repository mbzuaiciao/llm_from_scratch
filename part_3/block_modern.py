import torch.nn as nn
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from attn_modern import CausalSelfAttentionModern

class TransformerBlockModern(nn.Module):
    """
    A modern Transformer block incorporating RMSNorm/LayerNorm, CausalSelfAttentionModern,
    and SwiGLU/GELU Feed-Forward Networks.
    This block supports various advanced features like RoPE, GQA, sliding window attention,
    and KV caching.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        """
        Initializes the TransformerBlockModern.
        - n_embd: The embedding dimension of the input.
        - n_head: The number of attention heads.
        - dropout: Dropout rate.
        - use_rmsnorm: If True, uses RMSNorm; otherwise, uses LayerNorm.
        - use_swiglu: If True, uses SwiGLU for the FFN; otherwise, uses GELU-based FFN.
        - rope: If True, enables Rotary Positional Embeddings in attention.
        - max_pos: Maximum sequence length for RoPE cache.
        - sliding_window: Size of the sliding window for attention. None for full attention.
        - attention_sink: Number of attention sink tokens.
        - n_kv_head: Number of K/V heads for Grouped-Query Attention (GQA).
                     If None, defaults to n_head (Multi-Head Attention).
        """
        super().__init__()
        # Choose normalization layer based on `use_rmsnorm` flag.
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        # First normalization layer before attention.
        self.ln1 = Norm(n_embd)
        # Causal Self-Attention module with modern features.
        self.attn = CausalSelfAttentionModern(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)
        
        # Second normalization layer before the feed-forward network.
        self.ln2 = Norm(n_embd)
        # Feed-Forward Network: SwiGLU or a standard GELU-based MLP.
        self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x, kv_cache=None, start_pos: int = 0):
        """
        Forward pass for the TransformerBlockModern.
        Applies pre-normalization, attention with residual connection, then FFN with residual connection.
        """
        # Apply pre-normalization, then attention. Add residual connection.
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a
        # Apply pre-normalization, then FFN. Add residual connection.
        x = x + self.ffn(self.ln2(x))
        return x, kv_cache