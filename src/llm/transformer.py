import torch
from torch import nn
from .mlp import FeedForwardLayer
from .attention import MultiHeadedAttention

class Transformer(nn.Module):
    def __init__(self,
        ctx_size: int = 1024,
        q_dim: int = 768,
        k_dim: int = 768,
        v_dim: int = 768,
        ffn_hidden_dim: int = 3072,
        dropout: float = 0.1,
        num_heads: int = 12,
        casual: bool = True,
        embedding_dim: int = 768,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.mha = MultiHeadedAttention(
            ctx_size=ctx_size,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            num_heads=num_heads,
            casual=casual,
            dropout=dropout,
            embedding_dim=embedding_dim
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardLayer(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x += self.mha(x)
        x = self.ln2(x)
        x += self.ffn(x)
        return x

if __name__ == '__main__':
    EMBEDDING_DIM = 768
    BATCH = 8
    SEQ_LEN = 512
    input = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    t = Transformer(embedding_dim=EMBEDDING_DIM)
    print(t(input))