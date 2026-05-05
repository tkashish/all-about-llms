import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Deque

import torch


@dataclass
class BlockParameters:
    num_tokens: int
    num_layers: int
    num_heads: int
    d_head: int
    dtype: torch.dtype
    device: str


class Block:
    def __init__(self, params: BlockParameters, block_id: int):
        self.k = torch.empty(params.num_layers, params.num_heads, params.num_tokens, params.d_head, dtype=params.dtype,
                             device=params.device)
        self.v = torch.empty(params.num_layers, params.num_heads, params.num_tokens, params.d_head, dtype=params.dtype,
                             device=params.device)
        self.tokens_used = 0
        self.max_num_tokens = params.num_tokens
        self.num_layers = params.num_layers
        self.num_heads = params.num_heads
        self.d_head = params.d_head
        self.block_id = block_id
        self.token_per_layer = {}

    # k and v shape: H T D [num_heads num_tokens d_head] for num_layers
    def add(self, k, v, layer_id):
        if not self.has_space():
            raise Exception(f"Block [{self.block_id}] is full")
        self.k[layer_id, :, max(self.tokens_used - 1, 0):k.shape[-2], :].copy_(k)
        self.v[layer_id, :, max(self.tokens_used - 1, 0):k.shape[-2], :].copy_(v)
        self.token_per_layer[layer_id] = k.shape[-2]
        if layer_id == self.num_layers - 1:
            self.tokens_used = k.shape[-2]

    def append(self, k: torch.Tensor, v: torch.Tensor, layer_id: int):
        if not self.has_space():
            raise Exception(f"Block [{self.block_id}] is full")
        self.k[layer_id, :, self.tokens_used:self.tokens_used + 1, :].copy_(k)
        self.v[layer_id, :, self.tokens_used:self.tokens_used + 1, :].copy_(v)
        self.token_per_layer[layer_id] = self.token_per_layer.get(layer_id, 0) + k.shape[-2]
        if layer_id == self.num_layers - 1:
            self.tokens_used += k.shape[-2]

    def get(self, layer_id) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_id, :, :self.token_per_layer[layer_id], :], self.v[
            layer_id, :, :self.token_per_layer[layer_id], :]

    def free(self):
        self.cache = torch.empty(self.max_num_tokens, self.num_layers, self.num_heads, self.d_head)

    def has_space(self) -> bool:
        return self.tokens_used < self.max_num_tokens

    def remaining_space(self) -> int:
        return self.max_num_tokens - self.tokens_used


class BlockPool:
    def __init__(self, params: BlockParameters, max_blocks: int):
        self.blocks: List[Block] = [Block(params, i) for i in range(max_blocks)]
        self.free_blocks: Deque[int] = deque()
        for i in range(max_blocks):
            self.free_blocks.append(i)
        self.lock = threading.RLock()
        self.max_tokens_per_block = params.num_tokens

    def new_block(self) -> Block:
        with self.lock:
            if not self.free_blocks:
                raise Exception(f"Pool is empty")
            index = self.free_blocks.popleft()
            return self.blocks[index]

    def release(self, block: Block):
        self.free_blocks.append(block.block_id)
        block.free()


class Sequence:
    def __init__(self, block_pool: BlockPool):
        self.block_pool = block_pool
        self.blocks = [block_pool.new_block()]
        self.max_tokens_per_block = block_pool.max_tokens_per_block

    def release_blocks(self):
        for block in self.blocks:
            self.block_pool.release(block)

    def add(self, k, v, layer_id):
        num_tokens = k.shape[-2]
        # print(f"\n adding {num_tokens} tokens {layer_id}")
        curr_block_index = self.starting_block_index(layer_id, num_tokens)
        curr_block = self.blocks[curr_block_index]
        token_start = 0
        token_end = min(curr_block.remaining_space(), num_tokens)
        while True:
            curr_k = k[:, token_start: token_end, :]
            curr_v = v[:, token_start: token_end, :]
            curr_block.add(curr_k, curr_v, layer_id)
            num_tokens -= (token_end - token_start)
            if num_tokens <= 0:
                return
            curr_block_index = self.next_block(curr_block_index, layer_id)
            curr_block = self.blocks[curr_block_index]
            token_start = token_end
            token_end = token_start + min(curr_block.remaining_space(), num_tokens)

    def append(self, k, v, layer_id):
        # print(f"\n appending token {layer_id}")
        curr_block_index = self.starting_block_index(layer_id, 1)
        curr_block = self.blocks[curr_block_index]
        curr_block.append(k, v, layer_id)

    def get(self, layer_id):
        k, v = self.blocks[0].get(layer_id)
        k_l = [k]
        v_l = [v]
        for i in range(1, len(self.blocks)):
            kc, vc = self.blocks[i].get(layer_id)
            k_l.append(kc)
            v_l.append(vc)
        k = torch.cat(k_l, dim=1)
        v = torch.cat(v_l, dim=1)
        return k, v

    def starting_block_index(self, layer_id, num_tokens: int):
        if layer_id == 0:
            if not self.blocks[-1].has_space():
                self.blocks.append(self.block_pool.new_block())
            return len(self.blocks) - 1
        else:
            tokens = max(num_tokens - self.blocks[-1].tokens_used, 0)
            starting_block_index = tokens // self.max_tokens_per_block
            return len(self.blocks) - 1 - starting_block_index

    def next_block(self, curr_index: int, layer_id: int):
        if layer_id == 0:
            self.blocks.append(self.block_pool.new_block())
            return len(self.blocks) - 1
        else:
            return curr_index + 1


if __name__ == '__main__':
    pool = BlockPool(BlockParameters(
        num_tokens=16,
        num_layers=2,
        num_heads=2,
        d_head=5,
        dtype=torch.float32,
        device="cpu"
    ), max_blocks=5)
    seq = Sequence(block_pool=pool)
    k = torch.randn(2, 30, 5)
    v = torch.randn(2, 30, 5)
    seq.add(k, v, 0)
    seq.add(k, v, 1)
    kc, vc = seq.get(0)
    print(torch.equal(k, kc))
    print(torch.equal(v, vc))
    for i in range(5):
        kn = torch.randn(2, 1, 5)
        vn = torch.randn(2, 1, 5)
        seq.append(kn, vn, 0)
        seq.append(kn, vn, 1)
        kc, vc = seq.get(0)
        k = torch.cat((k, kn), dim=1)
        v = torch.cat((v, vn), dim=1)
        print(torch.equal(k, kc))
        print(torch.equal(v, vc))
