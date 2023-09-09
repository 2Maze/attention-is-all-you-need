from torch import nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .embeddings import TransformerEmbedding
from .modules import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int,
                 dim_ff: int,
                 dropout: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()

        # First attn modules
        self.attn_layer = MultiHeadAttention(dim_model=dim_model, num_heads=num_heads, dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

        # Encoder attn modules
        self.enc_attn_layer = MultiHeadAttention(dim_model=dim_model, num_heads=num_heads, dropout=dropout)
        self.enc_attn_layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

        # FF modules
        self.ff_layer = FeedForward(dim_model=dim_model, dim_ff=dim_ff, dropout=dropout)
        self.ff_layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                memory: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor) -> tuple[Tensor, Tensor]:
        # apply self-attention
        x1, _ = self.attn_layer(x, x, x, trg_mask)  # (batch_size, target_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1))  # (batch_size, target_seq_len, d_model)

        # apply encoder-decoder attention
        # memory is the output from encoder block (encoder states)
        x1, attn = self.enc_attn_layer(x, memory, memory, src_mask)

        # x1: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1))  # (batch_size, target_seq_len, d_model)

        # apply position-wise feed-forward
        x1 = self.ff_layer(x)  # (batch_size, target_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.ff_layer_norm(x + self.dropout(x1))  # (batch_size, target_seq_len, d_model)

        # x: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return x, attn


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size: int,
                 dim_model: int,
                 num_layers: int,
                 num_heads: int,
                 dim_ff: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 60,
                 padding_idx: Optional[int] = 0):
        super(Decoder, self).__init__()

        self.embedding_layer = TransformerEmbedding(vocab_size=trg_vocab_size,
                                                    dim_model=dim_model,
                                                    dropout=dropout,
                                                    max_seq_len=max_seq_len,
                                                    padding_idx=padding_idx)
        self.decoder_layers = nn.ModuleList([DecoderLayer(dim_model=dim_model,
                                                          num_heads=num_heads,
                                                          dim_ff=dim_ff,
                                                          dropout=dropout) for _ in range(num_layers)])

    def forward(self, x: Tensor,
                memory: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding_layer(x)
        for dec_layer in self.decoder_layers:
            x, attn = dec_layer(x, memory, src_mask, trg_mask)  # (batch_size, source_seq_len, d_model)
        return x, attn