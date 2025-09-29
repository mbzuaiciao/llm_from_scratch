import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    A simpler and faster alternative to LayerNorm. It normalizes the activations
    by their root mean square, and then scales them with a learnable gain.
    It omits the re-centering (mean subtraction) step of LayerNorm.

    Formula: y = x * g / rms(x), where rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Initializes the RMSNorm layer.
        - dim: The feature dimension of the input tensor.
        - eps: A small value added for numerical stability to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        # `g` in the formula, a learnable per-feature scaling parameter.
        # Initialized to ones, so initially it has no effect.
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMSNorm to the input tensor `x`.
        - x: Input tensor of shape (..., dim).
        """
        # 1. Calculate the reciprocal of the root mean square.
        #    rsqrt is an efficient way to compute 1 / sqrt(x).
        rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 2. Normalize the input and apply the learnable scaling factor.
        return x * rrms * self.weight