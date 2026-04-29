from dataclasses import dataclass

import torch
from torch import nn

from transformer.attention import Attention, AttentionParams
from transformer.multi_layer_perceptron import MLP

class Transformer(nn.Module):
    def __init__(self, params: AttentionParams):
        super().__init__()
        self.attention = Attention(params)
        self.mlp = MLP(params.d_model, params.d_ff)
        self.rms_norm1 = nn.RMSNorm(params.d_model)
        self.rms_norm2 = nn.RMSNorm(params.d_model)

    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        x = x + self.attention(self.rms_norm1(x), pos)
        x = x + self.mlp(self.rms_norm2(x))
        return x