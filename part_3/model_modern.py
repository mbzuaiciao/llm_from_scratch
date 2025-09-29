from __future__ import annotations
import torch
import torch.nn as nn
from block_modern import TransformerBlockModern
from tokenizer import ByteTokenizer

# Get the absolute path to the folder that contains part_2 and part_3
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

class GPTModern(nn.Module):
    """
    A modern GPT-style Transformer model incorporating various architectural improvements
    from Part 3, such as RMSNorm, SwiGLU, RoPE, GQA, and KV caching.
    """
    def __init__(self, vocab_size: int = 256, block_size: int = 256,
                 n_layer: int=4, n_head: int=4, n_embd: int=256, dropout: float=0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True, rope: bool = True,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        """
        Initializes the GPTModern model.
        - vocab_size: The size of the vocabulary.
        - block_size: The maximum sequence length the model can handle.
        - n_layer: Number of Transformer blocks.
        - n_head: Number of attention heads in each block.
        - n_embd: Embedding dimension.
        - dropout: Dropout rate.
        - use_rmsnorm: If True, uses RMSNorm; otherwise, uses LayerNorm.
        - use_swiglu: If True, uses SwiGLU for FFNs; otherwise, uses GELU-based FFNs.
        - rope: If True, enables Rotary Positional Embeddings.
        - max_pos: Maximum sequence length for RoPE cache.
        - sliding_window: Size of the sliding window for attention.
        - attention_sink: Number of attention sink tokens.
        - n_kv_head: Number of K/V heads for Grouped-Query Attention (GQA).
        """
        super().__init__()
        self.block_size = block_size
        # Token embeddings: maps token IDs to embedding vectors.
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # Positional embeddings are handled by RoPE within the attention mechanism,
        # so a separate `pos_emb` layer is not needed here.
        self.drop = nn.Dropout(dropout)
        # Stack of modern Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlockModern(n_embd, n_head, dropout, use_rmsnorm, use_swiglu, rope, max_pos, sliding_window, attention_sink, n_kv_head)
            for _ in range(n_layer)
        ])
        # Final normalization layer. Uses Identity if RMSNorm is enabled (as RMSNorm is applied inside blocks).
        # Otherwise, uses LayerNorm.
        self.ln_f = nn.Identity() if use_rmsnorm else nn.LayerNorm(n_embd)
        # Output head: maps the final hidden states to vocabulary logits.
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache_list=None, start_pos: int = 0):
        """
        Forward pass for the GPTModern model.
        - idx: Input tensor of token IDs (B, T).
        - targets: Optional target tensor for loss calculation (B, T).
        - kv_cache_list: List of KVCache objects, one for each block, used during generation.
        - start_pos: The starting position for RoPE, typically 0 for prefill/training,
                     and the current sequence length for subsequent token generation.
        """
        B, T = idx.shape
        assert T <= self.block_size
        
        # Token embeddings.
        x = self.tok_emb(idx) 
        # Apply dropout to embeddings.
        x = self.drop(x)

        new_caches = []
        # Pass through each Transformer block.
        for i, blk in enumerate(self.blocks):
            # Retrieve KV cache for the current block if available.
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = blk(x, kv_cache=cache, start_pos=start_pos)
            new_caches.append(cache)
        x = self.ln_f(x)
        logits = self.head(x)

        # Calculate loss if targets are provided.
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches # Return logits, loss, and updated KV caches.

    @torch.no_grad()
    def generate(self, 
                 prompt: torch.Tensor, 
                 max_new_tokens=200, 
                 temperature=1.0, 
                 top_k=50, 
                 top_p=None,
                 eos_id=1, # addition from part 6 for early stopping
                 sliding_window: int | None = None, 
                 attention_sink: int = 0):
        """
        Generates text autoregressively using the model with KV caching.
        - prompt: Initial sequence of token IDs (B, T_prompt).
        - max_new_tokens: Maximum number of tokens to generate.
        - temperature: Sampling temperature.
        - top_k: Top-k sampling parameter.
        - top_p: Top-p (nucleus) sampling parameter.
        - eos_id: End-of-sequence token ID for early stopping.
        - sliding_window: Passed to attention for dynamic windowing.
        - attention_sink: Passed to attention for attention sink tokens.
        """
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt
        kvs = [None] * len(self.blocks) # Initialize empty KV caches for each block.

        for _ in range(max_new_tokens):
            # feed full prompt once; then only the last token
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[:, -1:]

            # absolute start position from cache length (0 on first step)
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

            # Forward pass, getting updated KV caches.
            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # addition from part 6 for early stopping
            if eos_id is not None:
                if (next_id == eos_id).all():
                    break

        return idx


    @torch.no_grad()
    def generate_nocache(self, prompt: torch.Tensor, max_new_tokens=200, temperature=1.0, top_k=50, top_p=None,
                sliding_window: int | None = None, attention_sink: int = 0):
        """
        Generates text autoregressively without using KV caching (recomputes full context each step).
        This is generally slower but useful for comparison or specific scenarios.
        - prompt: Initial sequence of token IDs (B, T_prompt).
        - max_new_tokens: Maximum number of tokens to generate.
        - temperature: Sampling temperature.
        - top_k: Top-k sampling parameter.
        - top_p: Top-p (nucleus) sampling parameter.
        """
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt

        for _ in range(max_new_tokens):
            # Always run a full forward pass over the (potentially cropped) context window.
            # always run a full forward over the cropped window, with NO cache
            idx_cond = idx[:, -self.block_size:]
            # absolute position of first token in the window (matches cached path)
            start_pos = idx.size(1) - idx_cond.size(1)

            # Forward pass without KV cache.
            logits, _, _ = self(idx_cond, kv_cache_list=None, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            topv, topi = torch.topk(probs, 10)
            print("top ids:", topi.tolist())
            print("top vs:", topv.tolist())
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx
