import math
import os
import pickle

import numpy as np
import torch
from torch.optim import AdamW

from tokenizer.tokenizer import Tokenizer
from transformer.model import Model, HyperParams

TEXT_PATH = "data/corpus/TinyStoriesV2-GPT4-train.txt"
MODEL_PATH = "data/model/model.pt"
vocab_size      = 10000
d_model         = 64
num_heads       = 4         # D_head = 16
num_blocks      = 2
max_seq_len     = 64
d_ff            = 128       # 2 * d_model — smaller FFN for speed

batch_size      = 16
seq_len         = 64
num_steps       = 500
tokenizer_data_size = 500_000_000

def tokenize_data(output_file):
    with open(f"data/tokenizer/tinystories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"data/tokenizer/tinystories_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read(tokenizer_data_size)
        ids = tok.encode(text)                    # list[int]
        arr = np.array(ids, dtype=np.uint16)      # (N,)
        arr.tofile(output_file) # save

def get_batch(tokens: np.ndarray, B: int, T: int, device: str):
    # pick B random starting positions
    starts = np.random.randint(0, len(tokens) - T, size=B)
    # slice T+1 tokens starting at each
    batch = np.stack([tokens[s : s + T + 1] for s in starts])   # (B, T+1)
    return torch.from_numpy(batch).long().to(device)

def get_lr(step, warmup_steps=500, max_steps=num_steps, max_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

if __name__ == '__main__':
    tokenizer_bin = "data/tokenizer/tinystories_tokens.bin"
    if not os.path.exists(tokenizer_bin):
        os.makedirs("data", exist_ok=True)
        tokenize_data(tokenizer_bin)


    tokens = np.fromfile(tokenizer_bin, dtype=np.uint16)
    print(f"loaded {len(tokens):,} tokens")
    model = Model(HyperParams(
        d_model=d_model,
        vocab_size=vocab_size,
        num_heads = num_heads,
        max_seq_len=max_seq_len,
        d_ff=d_ff,
        num_transformer_blocks=num_blocks
    ))
    model.to("mps")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = AdamW(model.parameters(), lr=get_lr(0))

    import time
    start = time.time()
    config = {
        "vocab_size": vocab_size, "d_model": d_model, "num_heads": num_heads,
        "num_blocks": num_blocks, "max_seq_len": max_seq_len, "d_ff": d_ff,
    }
    for step in range(num_steps):
        learning_rate = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = learning_rate
        batch = get_batch(tokens, B=batch_size, T=seq_len, device='mps')
        inputs  = batch[:, :-1] # without the last token
        targets = batch[:, 1:] # withput the first
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            elapsed = time.time() - start
            toks = max(step, 1) * batch_size * seq_len
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] step {step:5d} | loss {loss.item():.4f} | lr {learning_rate:.2e} | {toks/elapsed:,.0f} tok/s", flush=True)
        if step > 0 and step % 1000 == 0:
            torch.save({"state_dict": model.state_dict(), "config": config}, MODEL_PATH)

    torch.save({"state_dict": model.state_dict(), "config": config}, MODEL_PATH)