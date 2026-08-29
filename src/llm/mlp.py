import torch
from torch import nn

class FeedForwardLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim: int = 768,
        hidden_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.W_in = nn.Linear(embedding_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.W_out = nn.Linear(hidden_dim, embedding_dim)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # (B, T, E)
        return self.ffn_drop(self.W_out(self.gelu(self.W_in(x))))

if __name__ == '__main__':
    EMBEDDING_DIM = 768
    BATCH = 8
    SEQ_LEN = 512
    input = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    print(input.shape)
    ffn = FeedForwardLayer(embedding_dim=EMBEDDING_DIM)
    print(ffn(input))