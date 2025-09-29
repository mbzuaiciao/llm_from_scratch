import torch.nn as nn

class FeedForward(nn.Module):
    """1.5 FFN with expansion factor `mult`.

    Dimensions:
      input:     (B, T, d_model)
      inner:     (B, T, mult*d_model)
      output:    (B, T, d_model)

    `mult*d_model` means the hidden width is `mult` times larger than `d_model`.
    Typical values: mult=4 for GELU FFN in GPT-style blocks.
    """
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        """
        Initializes the Feed-Forward Network.
        - d_model: The input and output dimension.
        - mult: The expansion factor for the hidden layer.
        - dropout: The dropout rate.
        """
        super().__init__()
        # A standard two-layer MLP.
        self.net = nn.Sequential(
            # 1. Expansion layer: projects from d_model to a wider hidden dimension.
            nn.Linear(d_model, mult * d_model),
            # 2. Non-linearity.
            nn.GELU(),
            # 3. Projection layer: projects back from the hidden dimension to d_model.
            nn.Linear(mult * d_model, d_model),
            # 4. Dropout for regularization.
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Applies the feed-forward network to the input tensor."""
        return self.net(x)