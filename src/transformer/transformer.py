import torch
from torch import nn

from transformer.attention import Attention


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = Attention(d_model, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)