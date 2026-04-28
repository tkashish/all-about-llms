from dataclasses import dataclass

import torch
from torch import nn

from transformer.embedding_table import EmbeddingTable
from transformer.transformer import Transformer


@dataclass
class HyperParams:
    d_model: int
    vocab_size: int
    num_heads: int

class Model(nn.Module):
    def __init__(self, params: HyperParams):
        super().__init__()
        self.embeddings_table = EmbeddingTable(params.d_model, params.vocab_size)
        self.transformer = Transformer(params.d_model, params.num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings_table(x)
        x = self.transformer(x)
        return x


if __name__ == '__main__':
    params =HyperParams(
        d_model=10,
        vocab_size=1000,
        num_heads = 2
    )
    model = Model(params)
    model.to("mps")
    input = torch.arange(0,12).reshape(4,3)
    input.to("mps")
    out = model(input)
    print(f"Input Shape: {input.shape}")
    print(f"Output Shape: {out.shape}")