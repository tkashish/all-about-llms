"""Benchmark inference speed: with KV cache vs without.

Usage:
    uv run python -m transformer.benchmark_inference
"""

import pickle
import time

import torch

from tokenizer.tokenizer import Tokenizer
from transformer.model import Model, HyperParams


MODEL_PATH = "data/model/model.pt"
VOCAB_PATH = "data/tokenizer/tinystories_vocab.pkl"
MERGES_PATH = "data/tokenizer/tinystories_merges.pkl"

PROMPT = "Once upon a time, there was a little girl"
NUM_NEW_TOKENS = 50
NUM_WARMUP_TOKENS = 0
DEBUG = True       # set True to print per-step K vectors for comparison
DEBUG_STEPS = 5    # how many decode steps to inspect


def load_model(config, state_dict, is_training: bool) -> Model:
    """Load a model instance — is_training=True disables the KV cache."""
    model = Model(HyperParams(
        d_model=config["d_model"],
        vocab_size=config["vocab_size"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        d_ff=config["d_ff"],
        num_transformer_blocks=config["num_blocks"],
        is_training=is_training,
    ))
    model.load_state_dict(state_dict)
    model.to("mps")
    model.eval()
    return model


def sync():
    """Force MPS to finish pending ops before we measure time."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


@torch.no_grad()
def generate_no_cache(model, prompt_ids, num_new, max_seq_len):
    """Naive O(N^2) generation: re-run forward on full sequence each step."""
    input_ids = prompt_ids.clone()
    per_step_times = []

    for i in range(num_new):
        sync()
        t0 = time.time()

        # truncate left if exceeded max_seq_len
        ctx = input_ids[:, -max_seq_len:]
        logits = model(ctx, -1)
        last_logits = logits[:, -1, :]
        next_id = torch.argmax(last_logits, dim=-1, keepdim=True)

        sync()
        per_step_times.append(time.time() - t0)

        if DEBUG and i < DEBUG_STEPS:
            top5 = torch.topk(last_logits[0], 5)
            print(f"  [no-cache step {i}] next_id={next_id.item()}  "
                  f"top5_vals={[round(v.item(), 3) for v in top5.values]}  "
                  f"top5_ids={top5.indices.tolist()}")

        input_ids = torch.cat([input_ids, next_id], dim=-1)

    return input_ids[:, prompt_ids.shape[-1]:], per_step_times


@torch.no_grad()
def generate_with_cache(model, prompt_ids, num_new, max_seq_len):
    """KV-cache generation: prefill once, then decode one token at a time."""
    generated = []
    per_step_times = []

    # prefill
    sync()
    t_prefill_start = time.time()
    logits = model(prompt_ids, -1)
    last_logits = logits[:, -1, :]
    next_id = torch.argmax(last_logits, dim=-1, keepdim=True)
    sync()
    prefill_time = time.time() - t_prefill_start

    if DEBUG:
        top5 = torch.topk(last_logits[0], 5)
        print(f"  [w-cache prefill] next_id={next_id.item()}  "
              f"top5_vals={[round(v.item(), 3) for v in top5.values]}  "
              f"top5_ids={top5.indices.tolist()}")

    generated.append(next_id)
    pos = prompt_ids.shape[-1]

    # decode
    for i in range(num_new - 1):
        sync()
        t0 = time.time()
        logits = model(next_id, pos)
        last_logits = logits[:, -1, :]
        next_id = torch.argmax(last_logits, dim=-1, keepdim=True)
        sync()
        per_step_times.append(time.time() - t0)

        if DEBUG and i < DEBUG_STEPS:
            top5 = torch.topk(last_logits[0], 5)
            print(f"  [w-cache step {i + 1}] pos={pos} next_id={next_id.item()}  "
                  f"top5_vals={[round(v.item(), 3) for v in top5.values]}  "
                  f"top5_ids={top5.indices.tolist()}")

        generated.append(next_id)
        pos += 1
        if pos >= max_seq_len:
            break

    return torch.cat(generated, dim=-1), prefill_time, per_step_times


def main():
    # load checkpoint + tokenizer
    ckpt = torch.load(MODEL_PATH)
    config = ckpt["config"]
    state_dict = ckpt["state_dict"]

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    prompt_ids = torch.tensor(
        [tok.encode(PROMPT)], dtype=torch.long, device="mps"
    )
    print(f"Prompt: {PROMPT!r}")
    print(f"Prompt tokens: {prompt_ids.shape[-1]}")
    print(f"Generating {NUM_NEW_TOKENS} new tokens (argmax sampling)")
    print()

    # ---- no cache ----
    print("Loading model WITHOUT KV cache (is_training=True path)...")
    model_nc = load_model(config, state_dict, is_training=True)

    # warmup (MPS compiles kernels on first run — don't count that)
    print("Warming up...")
    _ = generate_no_cache(model_nc, prompt_ids, NUM_WARMUP_TOKENS, config["max_seq_len"])

    print("Benchmarking no-cache...")
    sync()
    t0 = time.time()
    tokens_nc, times_nc = generate_no_cache(
        model_nc, prompt_ids, NUM_NEW_TOKENS, config["max_seq_len"]
    )
    sync()
    total_nc = time.time() - t0

    # free model before loading next
    del model_nc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- with cache ----
    print("\nLoading model WITH KV cache (is_training=False path)...")
    model_c = load_model(config, state_dict, is_training=False)

    print("Warming up...")
    _ = generate_with_cache(model_c, prompt_ids, NUM_WARMUP_TOKENS, config["max_seq_len"])

    print("Benchmarking with-cache...")
    sync()
    t0 = time.time()
    tokens_c, prefill_c, decode_times_c = generate_with_cache(
        model_c, prompt_ids, NUM_NEW_TOKENS, config["max_seq_len"]
    )
    sync()
    total_c = time.time() - t0

    # ---- report ----
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nNo cache:")
    print(f"  Total:    {total_nc:.2f} s")
    print(f"  Per tok:  {total_nc / NUM_NEW_TOKENS * 1000:.1f} ms")
    print(f"  Tok/s:    {NUM_NEW_TOKENS / total_nc:.1f}")
    print(f"  First step: {times_nc[0] * 1000:.1f} ms")
    print(f"  Last step:  {times_nc[-1] * 1000:.1f} ms")

    print(f"\nWith cache:")
    print(f"  Total:    {total_c:.2f} s")
    print(f"    Prefill:  {prefill_c * 1000:.1f} ms")
    print(f"    Decode:   {total_c - prefill_c:.2f} s")
    print(f"  Per decode tok: {(total_c - prefill_c) / (NUM_NEW_TOKENS - 1) * 1000:.1f} ms")
    print(f"  Tok/s (incl. prefill): {NUM_NEW_TOKENS / total_c:.1f}")
    if decode_times_c:
        print(f"  First decode step: {decode_times_c[0] * 1000:.1f} ms")
        print(f"  Last decode step:  {decode_times_c[-1] * 1000:.1f} ms")

    print(f"\nSpeedup: {total_nc / total_c:.1f}x")

    # correctness: argmax should match exactly
    match = torch.equal(tokens_nc, tokens_c)
    print(f"\nCorrectness (argmax outputs match): {match}")
    if not match:
        # find the first divergence
        min_len = min(tokens_nc.shape[-1], tokens_c.shape[-1])
        for i in range(min_len):
            if tokens_nc[0, i].item() != tokens_c[0, i].item():
                print(f"  First divergence at step {i}:")
                print(f"    no-cache:   {tokens_nc[0, i].item()}")
                print(f"    with-cache: {tokens_c[0, i].item()}")
                break

    print(f"\nSample output (no cache):    {tok.decode(tokens_nc[0].tolist())!r}")
    print(f"Sample output (with cache):  {tok.decode(tokens_c[0].tolist())!r}")


if __name__ == "__main__":
    main()
