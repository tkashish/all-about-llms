import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len):
        super().__init__()
        # RoPE
        t = torch.arange(max_seq_len, dtype=torch.float32)
        f = 1 /10000**(2*torch.arange(d_head // 2, dtype=torch.float32)/d_head)
        angles = t[:, None]*f[None, :]
        self.register_buffer("cos_table", torch.cos(angles))
        self.register_buffer("sin_table", torch.sin(angles))

    """
        pos will be -1 for training or first token generation during inference
        if pos < 0, we will get full B H T D vector will all the tokens
        During inference, we will get pos > 0 and only get tensor for the last token at [pos]
    """
    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        mid = x.shape[-1]//2
        if pos < 0:
            num_tokens = x.shape[-2]
            first = x[..., :mid] * self.cos_table[:num_tokens, :] - x[..., mid:] * self.sin_table[:num_tokens, :]
            second = x[..., :mid] * self.sin_table[:num_tokens, :] + x[..., mid:] * self.cos_table[:num_tokens, :]
        else:
            first = x[..., :mid] * self.cos_table[pos, :] - x[..., mid:] * self.sin_table[pos, :]
            second = x[..., :mid] * self.sin_table[pos, :] + x[..., mid:] * self.cos_table[pos, :]

        return torch.cat([first, second], dim=-1)