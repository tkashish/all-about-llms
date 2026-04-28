from dataclasses import dataclass

import torch
from torch import nn

from transformer.attention import Attention
from transformer.multi_layer_perceptron import MLP

@dataclass
class TransformerHyperParams:
    d_model: int
    num_heads: int
    max_seq_len: int
    d_ff: int

class Transformer(nn.Module):
    def __init__(self, params: TransformerHyperParams):
        super().__init__()
        self.attention = Attention(params.d_model, params.num_heads, params.max_seq_len)
        self.mlp = MLP(params.d_model, params.d_ff)
        self.rms_norm1 = nn.RMSNorm(params.d_model)
        self.rms_norm2 = nn.RMSNorm(params.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.rms_norm1(x))
        x = x + self.mlp(self.rms_norm2(x))
        return x