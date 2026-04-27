import argparse
import dataclasses
import json
import pickle
from collections import Counter

from .pre_tokenizer import PreTokenizer
import cProfile, pstats

@dataclasses.dataclass
class TokenizerParameters:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

@dataclasses.dataclass
class TokenizerTrainingParameters:
    input_path: str
    special_tokens: list[str]
    vocab_size: int

class Counters:
    def __init__(self, word_frequency):
        self.pre_token_freq = self.get_pre_token_frequency_from_words(word_frequency)
        self.byte_pair_freq = Counter()
        self.byte_pair_to_pre_token_map = {}
        self.build_byte_pair_freq()

    def get_pre_token_frequency_from_words(self, word_frequency: Counter) -> Counter:
        counter = Counter()
        for word, count in word_frequency.items():
            byts = [bytes([b]) for b in word.encode()]
            curr_bytes_list = []
            for i in range(len(byts)):
                curr_bytes_list.append(byts[i])
            counter[tuple(curr_bytes_list)] += count
        return counter

    def build_byte_pair_freq(self):
        for pre_token in self.pre_token_freq.keys():
            self.pre_token_to_byte_pair_count(pre_token)

    def update_pre_token(self, merged_bytes: tuple[bytes, bytes]):
        pre_tokens_to_update = self.byte_pair_to_pre_token_map[merged_bytes]
        del self.byte_pair_to_pre_token_map[merged_bytes]

        for byts in pre_tokens_to_update:
            count = self.pre_token_freq[byts]
            del self.pre_token_freq[byts]
            curr_bytes_list = []
            found_in_merge = False
            for i in range(len(byts) - 1):
                curr = byts[i] + byts[i + 1]
                self.decrement_byte_pair_count((byts[i], byts[i + 1]), count, byts)
                if found_in_merge:
                    found_in_merge = False
                    continue
                if (byts[i], byts[i + 1]) == merged_bytes:
                    found_in_merge = True
                    curr_bytes_list.append(curr)
                else:
                    curr_bytes_list.append(byts[i])
            if not found_in_merge:
                curr_bytes_list.append(byts[-1])
            pre_token = tuple(curr_bytes_list)
            self.pre_token_freq[pre_token] += count
            self.pre_token_to_byte_pair_count(pre_token)
        del self.byte_pair_freq[merged_bytes]

    def pre_token_to_byte_pair_count(self, pre_token):
        count = self.pre_token_freq[pre_token]
        for i in range(len(pre_token) - 1):
            tpl = (pre_token[i], pre_token[i+1])
            self.byte_pair_freq[tpl] += count
            tpl_set = self.byte_pair_to_pre_token_map.get(tpl, set())
            tpl_set.add(pre_token)
            self.byte_pair_to_pre_token_map[tpl]= tpl_set

    def decrement_byte_pair_count(self, byte_pair, count, pre_token):
        self.byte_pair_freq[byte_pair] -= count
        if self.byte_pair_freq[byte_pair] == 0:
            del self.byte_pair_freq[byte_pair]

        if byte_pair in self.byte_pair_to_pre_token_map:
            self.byte_pair_to_pre_token_map[byte_pair].discard(pre_token)
            if not self.byte_pair_to_pre_token_map[byte_pair]:
                del self.byte_pair_to_pre_token_map[byte_pair]

    def max_byte_pair(self):
        if len(self.byte_pair_freq) == 0:
            return None
        return max(self.byte_pair_freq.items(), key=lambda x: (x[1], x[0]))


class TokenizerTrainer:
    def __init__(self, params: TokenizerTrainingParameters):
        self.input_path = params.input_path
        self.vocab_size = params.vocab_size
        self.special_tokens = params.special_tokens

    def train_bpe(self):
        pre_tokenizer = PreTokenizer(self.input_path, self.special_tokens)
        word_freq = pre_tokenizer.process()
        vocab_index = 256
        vocab = {i: bytes([i]) for i in range(256)}
        for st in self.special_tokens:
            vocab[vocab_index] = st.encode()
            vocab_index += 1
        merge = []
        counters = Counters(word_freq)
        while len(vocab) < self.vocab_size:
            max_item = counters.max_byte_pair()
            if max_item is None:
                break
            vocab[vocab_index] = max_item[0][0] + max_item[0][1]
            merge.append(max_item[0])
            vocab_index += 1
            counters.update_pre_token(max_item[0])
        return vocab, merge

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to training corpus")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--out-prefix", required=True, help="Prefix for output files(e.g. 'tinystories')")
    args = p.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    trainer = TokenizerTrainer(TokenizerTrainingParameters(args.input, ["<|endoftext|>"], args.vocab_size))
    vocab, merges =trainer.train_bpe()
    profiler.disable()
    pstats.Stats(profiler).sort_stats("cumtime").print_stats(1000)
    with open(f"{args.out_prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{args.out_prefix}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
