"""1.2 Self-attention from first principles on a tiny example (NumPy only).
We use T=3 tokens, d_model=4, d_k=d_v=2, single-head.
This script prints intermediate tensors so you can trace the math.

Dimensions summary (single head)
--------------------------------
X:          (B=1, T=3, d_model=4)
Wq/Wk/Wv:   (d_model=4, d_k=2)
Q,K,V:      (1, 3, 2)
Scores:     (1, 3, 3)   = Q @ K^T
Weights:    (1, 3, 3)   = softmax over last dim
Output:     (1, 3, 2)   = Weights @ V
"""
import numpy as np

# Set print options for better readability of the output arrays.
np.set_printoptions(precision=4, suppress=True)

# --- 1. Setup: Toy inputs ---
# B=1 (batch size), T=3 (sequence length), d_model=4 (embedding dimension)
# Each row represents a token's embedding vector.
X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

# --- 2. Weight matrices ---
# In a real model, these are learned parameters. Here, we fix them for a deterministic example.
# d_k = d_v = 2 (dimension of key/query/value vectors)
# Wq, Wk, Wv project the input embeddings into the Q, K, V spaces.
Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)  # (d_model, d_k)
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)  # (d_model, d_k)
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)  # (d_model, d_v)

# --- 3. Project to Q, K, V ---
# Each token's embedding is projected to create its query, key, and value vectors.
Q = X @ Wq  # (1,3,2)
K = X @ Wk  # (1,3,2)
V = X @ Wv  # (1,3,2)

print("Q shape:", Q.shape, "\nQ=\n", Q[0])
print("K shape:", K.shape, "\nK=\n", K[0])
print("V shape:", V.shape, "\nV=\n", V[0])

# --- 4. Calculate attention scores (scaled dot-product) ---
# The dot product Q @ K^T measures the similarity between each query and all keys.
scale = 1.0 / np.sqrt(Q.shape[-1])
attn_scores = (Q @ K.transpose(0,2,1)) * scale  # (1,3,3)

# --- 5. Apply causal mask ---
# For a causal model, a token cannot attend to future tokens.
# We set the scores for future positions to a large negative number (-inf).
mask = np.triu(np.ones((1,3,3), dtype=bool), k=1)
attn_scores = np.where(mask, -1e9, attn_scores)

# --- 6. Compute attention weights (softmax) ---
# Softmax converts the scores into probabilities, representing how much attention
# each query should pay to each key.
weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
print("Weights shape:", weights.shape, "\nAttention Weights (causal)=\n", weights[0])

# --- 7. Compute the output (weighted sum of V) ---
# Each output token is a weighted sum of all value vectors, using the attention weights.
out = weights @ V  # (1,3,2)
print("Output shape:", out.shape, "\nOutput=\n", out[0])