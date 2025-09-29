import torch.nn as nn
from multi_head import MultiHeadSelfAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    """1.6 Transformer block = LN → MHA → residual → LN → FFN → residual."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        """
        Initializes a single Transformer block.
        - d_model: The embedding dimension.
        - n_head: The number of attention heads.
        - dropout: The dropout rate.
        """
        super().__init__()
        # First layer normalization (pre-normalization).
        self.ln1 = nn.LayerNorm(d_model)
        # Multi-Head Attention module.
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        # Second layer normalization (pre-normalization).
        self.ln2 = nn.LayerNorm(d_model)
        # Feed-Forward Network.
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(self, x):
        """
        Performs the forward pass for the Transformer block.
        This uses a "pre-norm" architecture.
        """
        # Residual connection around the attention block.
        x = x + self.attn(self.ln1(x))[0]
        # Residual connection around the feed-forward block.
        x = x + self.ffn(self.ln2(x))
        return x