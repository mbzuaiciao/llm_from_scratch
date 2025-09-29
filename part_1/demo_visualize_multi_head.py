"""Visualize multi-head attention weights per head (grid)."""
import torch
from multi_head import MultiHeadSelfAttention
from vis_utils import save_attention_heads_grid

# --- 1. Setup ---
# Define dimensions for a toy example.
# B=batch size, T=sequence length, d_model=embedding dimension, n_head=number of heads.
B, T, d_model, n_head = 1, 5, 12, 3
# Create a random input tensor representing a batch of sequences.
x = torch.randn(B, T, d_model)
# Instantiate the Multi-Head Attention module.
attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=False)

# --- 2. Forward Pass ---
# Run the input through the attention module to get the output and the attention weights.
out, w = attn(x)  # w has shape (B, H, T, T)

# --- 3. Visualization ---
# Save the attention weights as a grid of heatmaps, one for each head.
# We detach the tensor from the computation graph and move it to the CPU for NumPy conversion.
save_attention_heads_grid(w.detach().cpu().numpy(), filename="multi_head_attn_grid.png")