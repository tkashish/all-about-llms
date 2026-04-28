from dataclasses import dataclass

import einops
import torch
from torch import nn

from transformer.embedding_table import EmbeddingTable
from transformer.transformer import Transformer, TransformerHyperParams


@dataclass
class HyperParams:
    d_model: int
    vocab_size: int
    num_heads: int
    max_seq_len: int
    d_ff: int
    num_transformer_blocks: int

class Model(nn.Module):
    def __init__(self, params: HyperParams):
        super().__init__()
        self.embeddings_table = EmbeddingTable(params.d_model, params.vocab_size)
        self.transformers = nn.ModuleList([Transformer(TransformerHyperParams(
            d_model=params.d_model,
            num_heads =params.num_heads,
            max_seq_len=params.max_seq_len,
            d_ff=params.d_ff
        )) for _ in range(params.num_transformer_blocks)])
        self.rms_norm = nn.RMSNorm(params.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings_table(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.rms_norm(x)
        logits = einops.einsum(x, self.embeddings_table.table.T, "b t d, d t_all -> b t t_all")
        return logits

if __name__ == '__main__':
    params = HyperParams(
        d_model=20,
        vocab_size=1000,
        num_heads = 2,
        max_seq_len=10,
        d_ff=40,
        num_transformer_blocks=10
    )
    model = Model(params)
    model.to("mps")
    input = torch.arange(0,12).reshape(4,3)
    input.to("mps")
    out = model(input)
    print(f"Input Shape: {input.shape}")
    print(f"Output Shape: {out.shape}")
    print(out)