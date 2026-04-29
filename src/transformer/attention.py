import math
from dataclasses import dataclass

import einops
import torch
from torch import nn

from transformer.kv_cache import KVCache
from transformer.rope import RoPE


@dataclass
class AttentionParams:
    d_model: int
    num_heads: int
    max_seq_len: int
    d_ff: int
    kv_cache: KVCache | None
    layer_idx: int

class Attention(nn.Module):
    def __init__(self, params: AttentionParams):
        super().__init__()
        self.q_out = nn.Parameter(torch.randn(params.d_model, params.d_model)/math.sqrt(params.d_model))
        self.k_out = nn.Parameter(torch.randn(params.d_model, params.d_model)/math.sqrt(params.d_model))
        self.v_out = nn.Parameter(torch.randn(params.d_model, params.d_model)/math.sqrt(params.d_model))
        self.w_out = nn.Parameter(torch.randn(params.d_model, params.d_model)/math.sqrt(params.d_model))
        self.d_model = params.d_model
        self.num_heads = params.num_heads
        self.d_head = params.d_model // params.num_heads
        causal_mask = torch.triu(torch.ones(params.max_seq_len, params.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)
        self.rope = RoPE(d_head=self.d_head, max_seq_len=params.max_seq_len)
        self.kv_cache = params.kv_cache
        self.layer_idx = params.layer_idx

    def rearrange_for_multi_head(self, tsr: torch.Tensor ):
        tsr = einops.rearrange(tsr, "b t (h d) -> b t h d", h = self.num_heads)
        return einops.rearrange(tsr, "b t h d -> b h t d")

    """
    Multi-Head Attention Layer
    pos will be < 0 for training or when generating first token during inference
    Training: we need to generate full Q K V to train W_Q, W_K, W_V
    Inference:
        1. Generating 1st token: We will generate Q, K, V for all the tokens since we have multiple layers 
        to pass through and KV Cache is not initialized. 
            Q: Why not generate Q for only last token?
            A: Because K, V generation in subsequent layers need full Q and not just the last token
                                        
        2. From 2nd token onwards: We will have KV Cache populated and would only need to generate K V 
        for last token. K, V for previous tokens will be fetched from cache.
    """
    def forward(self, x: torch.Tensor, pos: int, is_training=True) -> torch.Tensor:
        q = einops.einsum(x, self.q_out, "b t d_in, d_in d_out -> b t d_out")
        q = self.rope(self.rearrange_for_multi_head(q), pos)

        k = einops.einsum(x, self.k_out, "b t d_in, d_in d_out -> b t d_out")
        k = self.rope(self.rearrange_for_multi_head(k), pos)

        v = einops.einsum(x, self.v_out, "b t d_in, d_in d_out -> b t d_out")
        v = self.rearrange_for_multi_head(v)

        if not is_training:
            """
                KVCache is only used during inference
            """
            if pos < 0:
                assert self.kv_cache
                self.kv_cache.add(self.layer_idx, k, v)
            else:
                assert self.kv_cache
                self.kv_cache.append(self.layer_idx, k, v)
                k, v = self.kv_cache.get(self.layer_idx)

        k_t = einops.rearrange(k, "b h t d -> b h d t")
        """
        We are going to calculate the weights for each head and then merge all heads 
        for calculating the final vector to be multipled with value tensor
        """
        score = einops.einsum(q, k_t, "b h t_q d, b h d t_k -> b h t_q t_k")
        # scaling by the size of d_head i.e. d_model / number_of_heads
        scaled_score = score / math.sqrt(self.d_head)
        token_len = x.shape[-2]
        scaled_score = scaled_score.masked_fill(self.causal_mask[:token_len, :token_len], float("-inf"))
        weights = torch.softmax(scaled_score, dim=-1)
        """
        merging back the weights for each head
        """
        context_aware_vector = einops.einsum(weights, v, "b h tq tk, b h tk d -> b h tq d")
        context_aware_vector = einops.rearrange(context_aware_vector, "b h t d -> b t h d")
        context_aware_vector = einops.rearrange(context_aware_vector, "b t h d -> b t (h d)", h=self.num_heads)
        return einops.einsum(context_aware_vector, self.w_out, "b t d_in, d_in d_out -> b t d_out")