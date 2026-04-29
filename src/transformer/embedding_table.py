import math

import torch
from torch import nn


class EmbeddingTable(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        # dividing by sqrt of d_model keeps the variance after matmul roughly 1.
        # after matmul we do multiplication and addition which would have increase variance.

    """
    B = Batch Size
    T = Tokens per batch
    D = Tensor per token
    Input: 
    x.shape = [B, T]
    Output:
    out.shape =  [B, T, D]
    In this layer we want to add the vector for each token
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x]

