from .transformer import Transformer
from torch import nn
import torch

class GPT(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ctx_size: int = 1024,
        q_dim: int = 768,
        k_dim: int = 768,
        v_dim: int = 768,
        ffn_hidden_dim: int = 3072,
        num_heads: int = 12,
        num_blocks: int = 12,
        dropout: float = 0.1,
        vocab_size: int = 50627
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(ctx_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.lnf = nn.LayerNorm(embedding_dim)

        self.blocks = nn.Sequential(
            *[
                Transformer(
                    embedding_dim=embedding_dim,
                    ctx_size=ctx_size,
                    q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, 
                    dropout=dropout, num_heads=num_heads, casual=True,
                    ffn_hidden_dim=ffn_hidden_dim
                ) for _ in range (0, num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor):
        _, T = x.shape
        x = self.token_embedding(x)

        positions = torch.arange(T, device=x.device)

        x += self.pos_embedding(positions)

        x = self.blocks(x)

        x = self.lnf(x)

        x = self.lm_head(x)

        return x

