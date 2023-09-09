import torch
import math

from torch import nn
from torch import Tensor
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim_model: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 60) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # compute positional encodings
        pe = torch.zeros(max_seq_len, dim_model)  # (max_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )  # (d_model,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        """
            x: (batch_size, seq_len, dim_model)
        """
        x = x + self.pe[:x.size(0), :]  # (batch_size, seq_len, dim_model)
        return self.dropout(x)  # (batch_size, seq_len, dim_model)


class TransformerEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim_model: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 60,
                 padding_idx: Optional[int] = 0) -> None:
        super(TransformerEmbedding, self).__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=dim_model,
                                       padding_idx=padding_idx)
        self.positional_encoder = PositionalEncoding(dim_model=dim_model,
                                                     dropout=dropout,
                                                     max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        return self.positional_encoder(self.embeddings(x))