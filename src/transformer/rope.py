import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len):
        super().__init__()
        # RoPE
        t = torch.arange(max_seq_len, dtype=torch.float32)
        f = 1 /10000**(2*torch.arange(d_head // 2, dtype=torch.float32)/d_head)
        angles = t[:, None]*f[None, :]
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)
        self.register_buffer("cos", cos_table)
        self.register_buffer("sin", sin_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = x.shape[-1]//2
        num_tokens = x.shape[-2]
        first = x[..., :mid] * self.cos[:num_tokens, :] - x[..., mid:] * self.sin[:num_tokens, :]
        second = x[..., :mid] * self.sin[:num_tokens, :] + x[..., mid:] * self.cos[:num_tokens, :]
        return torch.cat([first, second], dim=-1)