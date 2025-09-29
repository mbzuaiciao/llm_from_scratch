"""1.1 Positional encodings (absolute learned + sinusoidal)."""
import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """A positional encoding layer that learns a unique embedding for each position."""
    def __init__(self, max_len: int, d_model: int):
        """
        Initializes the learned positional encoding layer.
        - max_len: The maximum sequence length to create embeddings for.
        - d_model: The dimension of the model's embeddings.
        """
        super().__init__()
        # Create an embedding layer where each position index (0 to max_len-1)
        # is mapped to a learnable vector of size d_model.
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        """
        Adds learned positional embeddings to the input tensor.
        - x: Input tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        # Create a tensor of position indices [0, 1, ..., T-1].
        pos = torch.arange(T, device=x.device)
        # Look up the embeddings for these positions.
        pos_emb = self.emb(pos)  # (T, d_model)
        # Add the positional embeddings to the input tensor.
        # `pos_emb` is broadcasted across the batch dimension.
        return x + pos_emb.unsqueeze(0)  # broadcast over batch

class SinusoidalPositionalEncoding(nn.Module):
    """A positional encoding layer using sine and cosine functions of different frequencies."""
    def __init__(self, max_len: int, d_model: int):
        """
        Initializes the sinusoidal positional encoding layer.
        - max_len: The maximum sequence length to precompute encodings for.
        - d_model: The dimension of the model's embeddings.
        """
        super().__init__()
        # Create a matrix to hold the precomputed positional encodings.
        pe = torch.zeros(max_len, d_model)
        # Create a tensor of positions [0, 1, ..., max_len-1].
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term for the frequencies.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices in the d_model dimension.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the d_model dimension.
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register `pe` as a buffer so it's part of the model's state but not a parameter.
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        """Adds sinusoidal positional embeddings to the input tensor."""
        B, T, _ = x.shape
        # Add the precomputed encodings for the first T positions to the input tensor.
        return x + self.pe[:T].unsqueeze(0)