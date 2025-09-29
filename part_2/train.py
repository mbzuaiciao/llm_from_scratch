from __future__ import annotations
import argparse, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT


def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict:
    # Set the model to evaluation mode. This disables layers like dropout.
    model.eval()
    out = {}
    # Disable gradient calculations to save memory and computation, as we are not training.
    with torch.no_grad():
        # Iterate over both 'train' and 'val' splits.
        for split in ['train', 'val']:
            losses = []
            # Perform multiple iterations to get a statistically stable estimate of the loss.
            for _ in range(args.eval_iters):
                # Get a random batch of data from the specified split.
                xb, yb = ds.get_batch(split, args.batch_size, args.device)
                # Forward pass to get logits and loss.
                _, loss = model(xb, yb)
                losses.append(loss.item())
            # Calculate the average loss over all iterations.
            out[split] = sum(losses) / len(losses)
    # Set the model back to training mode.
    model.train()
    return out


def main():
    # --- Argument Parsing ---
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help="Path to the training data file.")
    p.add_argument('--out_dir', type=str, default='runs/min-gpt', help="Directory to save checkpoints.")
    p.add_argument('--block_size', type=int, default=256, help="Context window size.")
    p.add_argument('--batch_size', type=int, default=32, help="Number of sequences per batch.")
    p.add_argument('--n_layer', type=int, default=4, help="Number of transformer layers.")
    p.add_argument('--n_head', type=int, default=4, help="Number of attention heads.")
    p.add_argument('--n_embd', type=int, default=256, help="Embedding dimension.")
    p.add_argument('--dropout', type=float, default=0.0, help="Dropout rate.")
    p.add_argument('--steps', type=int, default=2000, help="Total number of training steps.")
    p.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    p.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay for AdamW optimizer.")
    p.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")
    p.add_argument('--eval_interval', type=int, default=200, help="How often to run evaluation.")
    p.add_argument('--eval_iters', type=int, default=50, help="Number of batches for loss estimation.")
    p.add_argument('--sample_every', type=int, default=200, help="How often to generate a sample text.")
    p.add_argument('--sample_tokens', type=int, default=256, help="Number of tokens to generate in a sample.")
    p.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature.")
    p.add_argument('--top_k', type=int, default=50, help="Top-k sampling.")
    p.add_argument('--top_p', type=float, default=None, help="Nucleus (top-p) sampling.")
    p.add_argument('--cpu', action='store_true', help="Force use of CPU.")
    p.add_argument('--compile', action='store_true', help="Use torch.compile for model optimization (PyTorch 2.0+).")
    p.add_argument('--amp', action='store_true', help="Use Automatic Mixed Precision (AMP) for training.")
    args = p.parse_args()

    # --- Device Setup ---
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # --- Initialization ---
    tok = ByteTokenizer()
    ds = ByteDataset(args.data, block_size=args.block_size)
    model = GPT(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(args.device)

    # --- Model Compilation (Optional) ---
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling the model...")
        model = torch.compile(model)

    # --- Optimizer and AMP Scaler ---
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    use_amp = args.amp and 'cuda' in args.device.type
    scaler = torch.amp.GradScaler(device=args.device, enabled=use_amp)

    # --- Checkpoint Config ---
    # Create the config dict once to be saved with checkpoints.
    config_to_save = {
        'vocab_size': tok.vocab_size,
        'block_size': args.block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'dropout': args.dropout,
    }
    # --- Training Loop ---
    best_val = float('inf')
    t0 = time.time()
    model.train()
    # Ensure output directory exists before the loop starts
    import os; os.makedirs(args.out_dir, exist_ok=True)

    for step in range(1, args.steps + 1):
        # Fetch a batch of training data.
        xb, yb = ds.get_batch('train', args.batch_size, args.device)

        # Forward and backward pass with Automatic Mixed Precision (AMP)
        with torch.amp.autocast(device_type=args.device.type, enabled=use_amp):
            _, loss = model(xb, yb)
        # Clear previous gradients.
        opt.zero_grad(set_to_none=True)
        # Scale the loss and perform backward pass.
        scaler.scale(loss).backward()
        # Gradient clipping to prevent exploding gradients.
        if args.grad_clip > 0:
            # Unscale gradients before clipping.
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # Update model weights.
        scaler.step(opt)
        # Update the scaler for the next iteration.
        scaler.update()

        # --- Logging ---
        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {(time.time()-t0):.1f}s")
            t0 = time.time()

        # --- Evaluation and Checkpointing ---
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
            # If validation loss improves, save the model checkpoint.
            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt_path = f"{args.out_dir}/model_best.pt"
                torch.save({'model': model.state_dict(), 'config': config_to_save}, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")

        # --- Sampling ---
        if args.sample_every > 0 and step % args.sample_every == 0:
            # Generate a sample text from the model to qualitatively check progress.
            start = torch.randint(low=0, high=len(ds.train) - args.block_size - 1, size=(1,)).item()
            seed = ds.train[start:start + args.block_size].unsqueeze(0).to(args.device)
            out = model.generate(seed, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            txt = tok.decode(out[0].cpu())
            print("\n================ SAMPLE ================\n" + txt[-(args.block_size + args.sample_tokens):] + "\n=======================================\n")

    # --- Final Save ---
    torch.save({'model': model.state_dict(), 'config': config_to_save}, f"{args.out_dir}/model_final.pt")
    print(f"Saved final model to {args.out_dir}/model_final.pt")


if __name__ == '__main__':
    main()