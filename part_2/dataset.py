from __future__ import annotations
from pathlib import Path
import torch

class ByteDataset:
    """Holds raw bytes of a text file and yields (x,y) blocks for LM.
    - block_size: sequence length (context window)
    - split: fraction for training (rest is val)
    """
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        # Read the entire text file as a sequence of raw bytes.
        byte_data = Path(path).read_bytes()
        # Convert the byte sequence into a 1D tensor of integers (0-255).
        full_tensor = torch.tensor(list(byte_data), dtype=torch.long)
        # Calculate the index at which to split the data into training and validation sets.
        n = int(len(full_tensor) * split)
        # Create the training and validation tensors by slicing the full tensor.
        self.train = full_tensor[:n]
        self.val = full_tensor[n:]
        # Store the block_size, which defines the length of a single sequence (context window).
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        # Select the appropriate data split (train or val).
        buf = self.train if which == 'train' else self.val
        # Ensure the selected data split is large enough to create at least one full block.
        if len(buf) <= self.block_size:
            raise ValueError(f"Dataset '{which}' split is too small for block_size={self.block_size}. "
                             f"Need at least {self.block_size + 1} bytes, but got {len(buf)}.")
        # Generate `batch_size` random starting indices for the sequences.
        # The upper bound ensures that we can always slice a full block for both x and y.
        ix = torch.randint(0, len(buf) - self.block_size, (batch_size,))
        # Create the input sequences (x) by slicing `block_size` chunks from the data
        # starting at each of the random indices.
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        # Create the target sequences (y) by slicing chunks that are shifted one position
        # to the right of the input sequences. This is for next-token prediction.
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
        # Move the input and target tensors to the specified device (e.g., 'cuda' or 'cpu').
        return x.to(device), y.to(device)