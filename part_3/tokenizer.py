from __future__ import annotations
import torch

class ByteTokenizer:
    """A simple, universal tokenizer that operates at the byte level.

    It encodes a string into a sequence of integers (0-255) corresponding to its
    UTF-8 byte representation. It is stateless and requires no training.
    """
    def encode(self, s: str) -> torch.Tensor:
        """
        Encodes a string into a tensor of its UTF-8 byte values.
        - s: The input string.
        """
        # 1. Convert the input string to a sequence of bytes using UTF-8 encoding.
        # 2. Convert the bytes object to a list of integers (0-255).
        # 3. Create a PyTorch LongTensor from the list of integers.
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)
    def decode(self, ids) -> str:
        """
        Decodes a list or tensor of byte values back into a string.
        - ids: A list or tensor of integers (0-255).
        """
        # 1. If the input is a PyTorch tensor, convert it to a Python list.
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # 2. Convert the list of integers back into a bytes object.
        # 3. Decode the bytes object into a UTF-8 string, ignoring any invalid byte sequences.
        return bytes(ids).decode('utf-8', errors='ignore')
    @property
    def vocab_size(self) -> int:
        # The vocabulary consists of all possible byte values, so the size is fixed at 256.
        return 256