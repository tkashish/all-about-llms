from typing import Iterator

import regex as re
import pickle

class WordEncoder:
    def __init__(self, vocab, merge):
        self.vocab_map = {}
        self.cache = {}
        for i, b in vocab.items():
            self.vocab_map[b] = i
            self.cache[b] = [i]
        self.merge_map = self._merge_order_map(merge)

    def _merge_order_map(self, merges):
        merge_map = {}
        for i, word in enumerate(merges):
            merge_map[word] = i
        return merge_map

    def _get_highest_byte_pair(self, byte_list):
        pair = None
        rank = float('inf')
        for i in range(len(byte_list) - 1):
            tup = (byte_list[i], byte_list[i + 1])
            if tup in self.merge_map:
                if rank > self.merge_map[tup]:
                    pair = tup
                    rank = self.merge_map[tup]
        return pair

    def tokenize(self, word):
        encoded_word = word.encode()
        if encoded_word in self.cache:
            return self.cache[encoded_word]
        byts = [bytes([b]) for b in encoded_word]
        merge_found = True
        while merge_found:
            merge_found = False
            byte_pair = self._get_highest_byte_pair(byts)
            curr_byts = []
            found_match = False
            for i in range(len(byts) - 1):
                if found_match:
                    found_match = False
                    continue
                tup = (byts[i], byts[i + 1])
                if tup == byte_pair:
                    curr_byts.append(byts[i] + byts[i + 1])
                    found_match = True
                    merge_found = True
                else:
                    curr_byts.append(byts[i])
            if not found_match:
                curr_byts.append(byts[-1])
            byts = curr_byts
        tokens = []
        for byt in byts:
            if byt in self.vocab_map:
                tokens.append(self.vocab_map[byt])
            else:
                tokens.append(byt[0])
        self.cache[word] = tokens
        return tokens

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.word_encoder = WordEncoder(vocab, merges)
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.special_token_set = set(self.special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return

    def encode(self, text: str) -> list[int]:
        # raw text → token IDs
        tokens = []
        chunks = self.split_special(text)
        for chunk in chunks:
            if chunk in self.special_token_set:
                tokens.extend(self.word_encoder.tokenize(chunk))
            else:
                for word in self.pre_tokenize(chunk):
                    tokens.extend(self.word_encoder.tokenize(word))
        return tokens

    def encode_iterable(self, iterable) -> Iterator[int]:
        for item in iterable:
            yield from self.encode(item)

    def split_special(self, text):
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        return re.split(pattern, text) if self.special_tokens else [text]

    def pre_tokenize(self, word):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return re.findall(PAT, word)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")


if __name__ == '__main__':
    with open("tinystories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("tinystories_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # roundtrip test
    text = "Hello, world! This is a story about a cat."
    print(text)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"ids: {ids}")
    print(decoded)
    print(f"len: {len(ids)} tokens, {len(text)} bytes")
    print(f"roundtrip match: {decoded == text}")