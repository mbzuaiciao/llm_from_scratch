# Repository layout (Part 3)
#
#   part_3/
#     orchestrator.py              # runs tests + a small generation demo
#     tokenizer.py                 # local byte-level tokenizer (self-contained)
#     rmsnorm.py                   # 3.1 RMSNorm
#     rope.py                      # 3.2 RoPE cache + apply
#     swiglu.py                    # 3.3 SwiGLU FFN
#     kv_cache.py                  # 3.4/3.6 KV cache + rolling buffer
#     attn_modern.py               # attention w/ RoPE, sliding window, sink, optional KV cache
#     block_modern.py              # block = (RMSNorm|LN) + modern attention + (SwiGLU|GELU)
#     model_modern.py              # GPTModern wrapper with feature flags
#     demo_generate.py             # simple generation demo (shows KV cache + sliding window)
#     tests/
#       test_rmsnorm.py
#       test_rope_apply.py
#       test_kvcache_shapes.py
#
# Run from inside `part_3/`:
#   cd part_3
#   python orchestrator.py --demo
#   pytest -q

import argparse, pathlib, subprocess, sys, shlex

# Get the root directory of the current script (part_3/).
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    """
    Executes a shell command in the script's directory and exits if it fails.
    - cmd: The command string to execute.
    """
    print(f"\n>>> {cmd}")
    # Use shlex.split to handle command-line arguments correctly.
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    # If the command returns a non-zero exit code, it indicates an error.
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Set up an argument parser to handle command-line flags.
    p = argparse.ArgumentParser(description="Run tests and an optional generation demo for Part 3.")
    # The --demo flag enables the generation demo.
    p.add_argument("--demo", action="store_true", help="run a tiny generation demo")
    args = p.parse_args()

    # --- Main Execution ---
    # 1) Always run the unit tests to verify the correctness of the modern components.
    # Test RMSNorm implementation.
    run("uv run pytest -q tests/test_rmsnorm.py")
    # Test RoPE implementation.
    run("uv run pytest -q tests/test_rope_apply.py")
    # Test KV cache shapes and logic.
    run("uv run pytest -q tests/test_kvcache_shapes.py")

    # 2) If the --demo flag is provided, run the generation demo script.
    # This showcases the new architectural features in action.
    if args.demo:
        # The demo script is run with flags to enable RMSNorm, RoPE, SwiGLU,
        # and sliding window attention with an attention sink.
        run("uv run python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")

    print("\nPart 3 checks complete. ✅")