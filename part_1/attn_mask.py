import torch

def causal_mask(T: int, device=None):
    """
    Creates a causal mask for self-attention mechanisms.

    In a causal (or autoregressive) model, a token at position `i` should only be
    able to attend to tokens at positions `j <= i`. This mask prevents attention
    to future tokens.

    Returns a bool mask where True means the position is *masked* (disallowed).
    The final shape is (1, 1, T, T), which is suitable for broadcasting with the
    attention scores tensor of shape (B, heads, T, T).
    """
    # 1. Create a square matrix of size T x T filled with ones.
    #    This represents all possible query-key attention pairs.
    ones = torch.ones((T, T), dtype=torch.bool, device=device)
    # 2. Get the upper triangular part of the matrix, excluding the main diagonal.
    #    `torch.triu` with `diagonal=1` sets all elements below the first
    #    super-diagonal to False. The resulting True values are the positions
    #    that should be masked (e.g., query at pos 0 attending to key at pos 1).
    m = torch.triu(ones, diagonal=1)
    # 3. Reshape the mask to (1, 1, T, T) for broadcasting.
    #    This allows the same mask to be applied across all batches and heads.
    return m.view(1, 1, T, T)