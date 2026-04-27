import os
from collections import Counter
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from typing import BinaryIO

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


class PreTokenizer:
    def __init__(self, input_path: str, special_tokens: list[str]):
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.num_processes = 8

    def get_chunks(self) -> list[str]:
        chunks = []
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunks.append((start, end))
        return chunks

    def read_chunk(self, start: int, end: int):
        with open(self.input_path, "rb") as f:
            f.seek(start)
            return f.read(end - start).decode("utf-8", errors="ignore")

    def process_chunks(self, start: int, end: int, process_id) -> Counter:
        chunk = self.read_chunk(start, end)
        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            sub_chunks = re.split(pattern, chunk)
        else:
            sub_chunks = [chunk]
        frequency_counter = Counter()
        for sub_chunk in sub_chunks:
            for match in re.finditer(PAT, sub_chunk):
                token = match.group()
                frequency_counter[token] += 1
        return frequency_counter

    def process(self) -> Counter:
        chunks = self.get_chunks()
        frequency = Counter()
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            for process_id, chunk in enumerate(chunks):
                futures.append(executor.submit(self.process_chunks, chunk[0], chunk[1], f"process-{process_id + 1}"))
            for f in as_completed(futures):
                frequency.update(f.result())
        return frequency
