from __future__ import annotations
import torch
import math

class RoPECache:
    """Precomputes cosine and sine frequencies for Rotary Positional Embeddings.
    This cache dynamically grows if it encounters a sequence longer than `max_pos`.
    """
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        """
        Initializes the RoPE cache.
        - head_dim: The dimension of each attention head. Must be even.
        - max_pos: The maximum sequence length to precompute for.
        - base: The base value for the frequency calculation.
        - device: The device to store the cache on.
        """
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        # Build the initial cache.
        self._build(max_pos)

    def get(self, positions: torch.Tensor):
        """
        Retrieves the cosine and sine values for the given positions.
        - positions: A 1D tensor of token positions.
        """
        # Ensure positions is a 1D tensor.
        if positions.dim() == 2:
            positions = positions[0]
        # Determine the maximum position required.
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        # If the required length exceeds the cached length, expand the cache.
        if need > self.max_pos:
            self._build(max(need, int(self.max_pos * 2)))
        # Fetch the precomputed values for the given positions.
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin
    
    def _build(self, max_pos: int):
        """(Re)builds the cosine and sine tables for a new maximum position."""
        self.max_pos = max_pos
        # Calculate the inverse frequencies. This is the 'theta' in the RoPE paper.
        # Shape: (head_dim / 2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        # Create a tensor of positions from 0 to max_pos - 1.
        t = torch.arange(max_pos, device=self.device).float()
        # Calculate the outer product of positions and inverse frequencies to get the angles.
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        # Precompute and cache the cosine and sine values.
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Positional Embedding to the input tensor.
    This function rotates pairs of features along the last dimension.
    - x: Input tensor of shape (B, H, T, D), where D is head_dim.
    - cos, sin: Precomputed cosine and sine values of shape (T, D/2).
    """
    assert x.size(-1) % 2 == 0
    # Reshape cos and sin to be broadcastable with the input tensor x.
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # Split the last dimension of x into two halves.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Apply the rotation matrix formula.
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    # Create an output tensor and fill it with the rotated values.
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out
