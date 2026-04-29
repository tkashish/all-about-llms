import pickle
import sys

import torch

from tokenizer.tokenizer import Tokenizer
from transformer.model import Model, HyperParams

TEXT_PATH = "data/corpus/TinyStoriesV2-GPT4-train.txt"

vocab_size      = 10000    # matches your tokenizer
num_new_tokens = 100
temperature = 0.8

MODEL_PATH = "data/model/model.pt"

def infer(model, tok, prompt, max_seq_len):
    eot_id = tok.encode("<|endoftext|>")[0]
    ids = tok.encode(prompt)                                      # list[int]
    input_ids = torch.tensor([ids], dtype=torch.long, device="mps")  # (1, T)
    pos = -1

    with torch.no_grad():
        for _ in range(num_new_tokens):
            logits = model(input_ids[:, -max_seq_len:], pos)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == eot_id:
                break
            # decode just this one token and print
            token_text = tok.decode([next_id.item()])
            print(token_text, end="", flush=True)
            if pos < 0:
                pos = input_ids.shape[-1] # keeping 0 indexed
            else:
                pos += 1
            input_ids = next_id
            if pos == max_seq_len:
                break
    print()   # final newline

if __name__ == '__main__':
    ckpt = torch.load(MODEL_PATH)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt["state_dict"]
    else:
        config = {"vocab_size": vocab_size}
        state_dict = ckpt

    model = Model(HyperParams(
        d_model=config["d_model"],
        vocab_size=config["vocab_size"],
        num_heads = config["num_heads"],
        max_seq_len=config["max_seq_len"],
        d_ff=config["d_ff"],
        num_transformer_blocks=config["num_blocks"],
        is_training=False
    ))
    model.load_state_dict(state_dict)
    model.to("mps")
    model.eval()

    with open(f"data/tokenizer/tinystories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"data/tokenizer/tinystories_merges.pkl", "rb") as f:
        merges = pickle.load(f)
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


    while True:
        prompt = input("Prompt: ")
        infer(model, tok, prompt, config["max_seq_len"])
