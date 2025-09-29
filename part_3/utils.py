from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    This function modifies the logits distribution to constrain the set of possible next tokens
    that can be sampled. It is used to control the randomness and quality of text generation.
    - logits: (B, vocab_size) tensor of logits from the model.
    - top_k: If set, keeps only the top k tokens with the highest probability.
    - top_p: If set, keeps the smallest set of tokens whose cumulative probability exceeds p.
    Returns filtered logits where tokens that are filtered out are set to -infinity.
    """
    B, V = logits.shape
    # Make a copy to avoid modifying the original logits tensor in place.
    filtered = logits.clone()

    # --- Top-k filtering ---
    # If top_k is specified and is less than the vocabulary size,
    # keep only the k most likely tokens.
    if top_k is not None and top_k < V:
        # Get the values of the top k logits. We don't need the indices.
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
        # The k-th highest logit value becomes the threshold.
        kth = topk_vals[:, -1].unsqueeze(-1)
        # Set all logits lower than the threshold to -infinity.
        # They will have a probability of 0 after softmax.
        filtered[filtered < kth] = float('-inf')

    # --- Top-p (nucleus) filtering ---
    # If top_p is specified, keep the smallest set of tokens whose cumulative probability exceeds p.
    if top_p is not None and 0 < top_p < 1.0:
        # Sort logits in descending order to work with probabilities.
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        # Convert sorted logits to probabilities.
        probs = torch.softmax(sorted_logits, dim=-1)
        # Calculate the cumulative sum of probabilities.
        cumsum = torch.cumsum(probs, dim=-1)
        # Create a mask for tokens to remove. These are tokens whose cumulative probability is
        # greater than top_p.
        mask = cumsum > top_p
        # An important edge case: make sure we always keep at least the most likely token.
        mask[..., 0] = False
        # Apply the mask to the sorted logits, setting the removed tokens to -infinity.
        sorted_logits[mask] = float('-inf')
        # --- Scatter back to original order ---
        # Create a tensor of -infinity and use scatter to place the filtered (and still sorted)
        # logits back into their original positions.
        filtered = torch.full_like(filtered, float('-inf'))
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered