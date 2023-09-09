import torch
import numpy as np

from torch import nn
from torch import Tensor


class Transformer(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 classifier: nn.Module,
                 pad_idx: int = 0):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.config = None

    def get_padding_mask(self, x: Tensor) -> Tensor:
        """ x: (batch_size, seq_len)
        """
        x = (x != self.pad_idx).unsqueeze(-2)  # (batch_size, 1, seq_len)
        return x

    def get_subsequent_mask(self, x: Tensor) -> Tensor:
        """ x: (batch_size, seq_len)
        """
        seq_len = x.size(1)
        subsequent_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype(np.int8)  # (batch_size, seq_len, seq_len)
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0).to(x.device)  # (batch_size, seq_len, seq_len)
        return subsequent_mask

    def forward(self,
                src: Tensor,
                trg: Tensor) -> tuple[Tensor, Tensor]:
        """ src: (batch_size, source_seq_len)
            tgt: (batch_size, target_seq_len)
        """
        # create masks for source and target
        src_mask = self.get_padding_mask(src)
        trg_mask = self.get_padding_mask(trg) & self.get_subsequent_mask(trg)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)

        # encode the source sequence
        enc_output = self.encoder(src, src_mask)  # (batch_size, source_seq_len, d_model)

        # decode based on source sequence and target sequence generated so far
        dec_output, attn = self.decoder(trg, enc_output, src_mask, trg_mask)

        # dec_output: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)

        # apply linear projection to obtain the output distribution
        output = self.classifier(dec_output)  # (batch_size, target_seq_len, vocab_size)

        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return output, attn