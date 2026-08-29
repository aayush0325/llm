import torch
import math
from torch import nn

class MultiHeadedAttention(nn.Module):
    def __init__(
            self, 
            ctx_size: int = 1024,
            embedding_dim: int = 768,
            q_dim: int = 768, 
            k_dim: int = 768, 
            v_dim: int = 768, 
            num_heads: int = 12,
            casual: bool = True,
            dropout: float = 0.1
        ):
        assert q_dim == k_dim
        assert q_dim % num_heads == 0

        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.ctx_size = ctx_size
        self.embedding_dim = embedding_dim

        self.W_q = nn.Linear(self.embedding_dim, q_dim) 
        self.W_k = nn.Linear(self.embedding_dim, k_dim)
        self.W_v = nn.Linear(self.embedding_dim, v_dim)
        self.W_out = nn.Linear(self.v_dim, self.embedding_dim)

        self.num_heads = num_heads
        self.casual = casual
        self.attn_drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(ctx_size, ctx_size), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        # x_dim (B, T, E)
        B, T, _ = x.shape

        q: torch.Tensor = self.W_q(x) # (B, T, q_dim)
        k: torch.Tensor = self.W_k(x) # (B, T, k_dim)
        v: torch.Tensor = self.W_v(x) # (B, T, v_dim)

        q = q.reshape(B, T, self.num_heads, self.q_dim // self.num_heads).transpose(1, 2) # (B, num_heads, T, q_head_dim)
        k = k.reshape(B, T, self.num_heads, self.k_dim // self.num_heads).transpose(1, 2) # (B, num_heads, T, k_head_dim)
        v = v.reshape(B, T, self.num_heads, self.v_dim // self.num_heads).transpose(1, 2) # (B, num_heads, T, v_head_dim)

        weights = q @ k.transpose(-1, -2) / math.sqrt(self.q_dim // self.num_heads) # (B, num_heads, T, T)

        if self.casual:
            weights = torch.masked_fill(weights, self.mask[:T, :T], float("-inf")) # masking (B, num_heads, T, T)

        scores = torch.softmax(weights, dim=-1) # (B, num_heads, T, T)

        values = scores @ v # (B, num_heads, T, v_dim / num_heads)

        values = values.transpose(1, 2).reshape(B, T, self.v_dim)

        out = self.W_out(values) # (B, T, E)

        return self.attn_drop(out)

if __name__ == '__main__':
    BATCH_SIZE = 8
    SEQ_LEN = 512
    EMBEDDING_DIM = 768

    input = torch.randn(8, 512, 768)

    attn = MultiHeadedAttention(embedding_dim=EMBEDDING_DIM)

    print(attn(input))