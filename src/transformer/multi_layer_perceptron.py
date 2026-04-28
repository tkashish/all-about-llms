import math
import torch
from einops import einops
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_signal = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model))
        self.w_gate = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model))
        self.w_o = nn.Parameter(torch.randn(d_ff, d_model) / math.sqrt(d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = einops.einsum(x, self.w_signal, "b t d, d d_ff -> b t d_ff")
        signal = F.silu(signal)
        gate = einops.einsum(x, self.w_gate, "b t d, d d_ff -> b t d_ff")
        x = signal*gate
        x = einops.einsum(x, self.w_o, "b t d_ff, d_ff d -> b t d")
        return x