import argparse, torch
from tokenizer import ByteTokenizer
from model_modern import GPTModern
import time

# This script demonstrates text generation using the GPTModern model.
# It showcases the performance difference between using a KV cache (`generate`)
# and not using one (`generate_nocache`).
#
# Example usage:
#   python demo_generate.py --rope --swiglu --rmsnorm --sliding_window 64 --sink 4

if __name__ == "__main__":
    # --- Argument Parsing ---
    p = argparse.ArgumentParser()
    p.add_argument('--rmsnorm', action='store_true', help="Use RMSNorm instead of LayerNorm.")
    p.add_argument('--rope', action='store_true', help="Use Rotary Positional Embeddings.")
    p.add_argument('--swiglu', action='store_true', help="Use SwiGLU for the feed-forward network.")
    p.add_argument('--sliding_window', type=int, default=None, help="Enable sliding window attention with this window size.")
    p.add_argument('--sink', type=int, default=0, help="Number of attention sink tokens to keep with sliding window.")
    p.add_argument('--group_size', type=int, default=2, help="Number of query heads per key/value head for GQA.")
    p.add_argument('--tokens', type=int, default=120, help="Number of tokens to generate.")
    p.add_argument('--cpu', action='store_true', help="Force use of CPU.")
    args = p.parse_args()

    # --- Device and Model Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Initialize a simple byte-level tokenizer.
    tok = ByteTokenizer()
    # Initialize the GPTModern model with the specified architectural features.
    model = GPTModern(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=4, n_embd=128,
                      use_rmsnorm=args.rmsnorm, use_swiglu=args.swiglu, rope=args.rope,
                      max_pos=4096, sliding_window=args.sliding_window, attention_sink=args.sink, n_kv_head=args.group_size).to(device)

    # Start generation from a newline character (token ID 10).
    prompt = torch.tensor([[10]], dtype=torch.long, device=device)

    # --- Generation ---
    with torch.no_grad():
        # Generate text using the efficient KV cache method and time it.
        start = time.time()
        out = model.generate(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"Generated {args.tokens} tokens in {time.time()-start:.2f} sec")

        # Generate text without the KV cache for comparison and time it.
        # This is expected to be significantly slower as it recomputes the full context each step.
        start = time.time()
        out_nocache = model.generate_nocache(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"(nocache) Generated {args.tokens} tokens in {time.time()-start:.2f} sec")
    
    # Decode and print the generated text from both methods.
    # The output should be identical, demonstrating functional equivalence.
    print(tok.decode(out[0].cpu()))
    print(tok.decode(out_nocache[0].cpu()))