from dataclasses import dataclass

import torch


@dataclass
class CacheParams:
    num_attention_layers: int
    max_seq_len: int
    d_head: int
    num_heads: int
    dtype: torch.dtype
    device: str


"""
Shape of K and V: B H T D
"""


class KVCache:
    def __init__(self, params: CacheParams):
        self.num_attention_layers = params.num_attention_layers
        self.pos = 0
        self.k = torch.ones(params.num_attention_layers,
                            1,
                            params.num_heads,
                            params.max_seq_len,
                            params.d_head, dtype=params.dtype)
        self.v = torch.ones(params.num_attention_layers,
                            1,
                            params.num_heads,
                            params.max_seq_len,
                            params.d_head, dtype=params.dtype)

    def get(self, layer_id):
        return self.k[layer_id, :, :, :self.pos+1, :], self.v[layer_id, :, :, :self.pos+1, :]

    def add(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.pos = k.shape[-2] - 1
        self.k[layer_id, :, :, :self.pos + 1, :].copy_(k)
        self.v[layer_id, :, :, :self.pos + 1, :].copy_(v)

    def append(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.pos += 1
        self.k[layer_id, :, :, :self.pos + 1, :].copy_(k)
        self.v[layer_id, :, :, :self.pos + 1, :].copy_(v)


# if __name__ == '__main__':
#     cache = KVCache(params=CacheParams(
#         num_attention_layers=2,
#         max_seq_len=10,
#         d_head=5,
#         num_heads=2,
#         dtype=torch.float32,
#         device="cpu"
#     ))
#     k = torch.randn(1, 2, 2, 5)
#     v = torch.randn(1, 2, 2, 5)
#     cache.add(0, k, v)
#     print(cache.get(0)[0].shape)
#     kn = torch.randn(1,2,1,5)
#     vn = torch.randn(1,2,1,5)
#     cache.append(0, kn, vn)
#     print(cache.get(0)[0].shape)
#     print(cache.get(0)[0])
