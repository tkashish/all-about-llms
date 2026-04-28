import math

import einops
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.q_out = nn.Parameter(torch.randn(d_model, d_model)/math.sqrt(d_model))
        self.k_out = nn.Parameter(torch.randn(d_model, d_model)/math.sqrt(d_model))
        self.v_out = nn.Parameter(torch.randn(d_model, d_model)/math.sqrt(d_model))
        self.w_out = nn.Parameter(torch.randn(max_seq_len, max_seq_len)/math.sqrt(d_model))
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def rearrange_for_multi_head(self, tsr: torch.Tensor ):
        tsr = einops.rearrange(tsr, "b t (h d) -> b t h d", h = self.num_heads)
        return einops.rearrange(tsr, "b t h d -> b h t d")

    """
    Multi-Head Attention Layer
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = einops.einsum(x, self.q_out, "b t d_in, d_in d_out -> b t d_out")
        k = einops.einsum(x, self.k_out, "b t d_in, d_in d_out -> b t d_out")
        v = einops.einsum(x, self.v_out, "b t d_in, d_in d_out -> b t d_out")

        """
        We are going to calculate the weights for each head and then merge all heads 
        for calculating the final vector to be multipled with value tensor
        """
        q = self.rearrange_for_multi_head(q)
        k = self.rearrange_for_multi_head(k)
        v = self.rearrange_for_multi_head(v)
        k_t = einops.rearrange(k, "b h t d -> b h d t")
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