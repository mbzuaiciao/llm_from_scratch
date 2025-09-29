import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    A variant of the Feed-Forward Network that uses a gated linear unit with
    the Swish activation function. It often provides better performance than
    standard ReLU or GELU-based FFNs.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        """
        Initializes the SwiGLU layer.
        - dim: The input and output dimension.
        - mult: The expansion factor for the hidden dimension.
        - dropout: The dropout rate.
        """
        super().__init__()
        # The hidden dimension is typically a multiple of the input dimension.
        inner = mult * dim
        # The first linear projection (W1 in the formula).
        self.w1 = nn.Linear(dim, inner, bias=False)
        # The second linear projection for the gate (W2 in the formula).
        self.w2 = nn.Linear(dim, inner, bias=False)
        # The final output projection (W3 in the formula).
        self.w3 = nn.Linear(inner, dim, bias=False)
        # The Swish activation function, which is equivalent to PyTorch's SiLU.
        self.act = nn.SiLU()
        # Dropout layer for regularization.
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        # Project the input to get the main branch 'a'.
        a = self.w1(x)
        # Project the input and apply the activation function to get the gate 'b'.
        b = self.act(self.w2(x))
        # Element-wise multiply the main branch by the gate, then project to the output.
        return self.drop(self.w3(a * b))