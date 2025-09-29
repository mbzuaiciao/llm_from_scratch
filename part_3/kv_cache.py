from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    """A simple container for the key and value tensors of the KV cache."""
    k: torch.Tensor  # Shape: (B, H, T, D)
    v: torch.Tensor  # Shape: (B, H, T, D)

    @property
    def T(self):
        """Returns the sequence length (T) of the cached tensors."""
        return self.k.size(2)

class RollingKV:
    """Implements a rolling buffer for the KV cache with an optional attention sink.
    This strategy keeps the first `sink` tokens and the last `window` tokens,
    discarding the tokens in between to manage memory usage for long sequences.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        Updates the cache with new key and value tensors and applies the rolling buffer logic.
        - k_new, v_new: The new key/value tensors to be added to the cache.
        """
        # On the first step, initialize the cache.
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            # Append the new key and value tensors to the existing cache.
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        
        # If the cache size exceeds the window + sink, apply the rolling buffer logic.
        if self.k.size(2) > self.window + self.sink:
            # Keep the 'sink' tokens at the beginning.
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            # Keep the most recent 'window' tokens from the end.
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            # Concatenate the sink and tail parts to form the new cache.
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        return self.k, self.v