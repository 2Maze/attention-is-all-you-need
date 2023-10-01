import torch
import torchmetrics
import torch.nn as nn
import math

from torchmetrics import Metric
from torch import optim
from types import ModuleType
from torch.nn import Transformer, TransformerEncoder
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self,
                 config) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(config.model['dropout'])

        # compute positional encodings
        pe = torch.zeros(5000, config.model['dim_model'])  # (max_len, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, config.model['dim_model'], 2).float() * (-math.log(10000.0) / config.model['dim_model'])
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


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_src = torch.load(config.vocab['path_src'])
        vocab_trg = torch.load(config.vocab['path_trg'])

        self.src_vocab = len(vocab_src)
        self.trg_vocab = len(vocab_trg)

        self.transformer = Transformer(d_model=config.model['dim_model'],
                                       nhead=config.model['num_heads'],
                                       num_encoder_layers=config.model['num_layers'],
                                       num_decoder_layers=config.model['num_layers'],
                                       dim_feedforward=config.model['dim_ff'],
                                       dropout=config.model['dropout'],
                                       batch_first=True)
        self.linear = nn.Linear(config.model['dim_model'], self.trg_vocab)
        self.src_tok_emb = TokenEmbedding(self.src_vocab, config.model['dim_model'])
        self.trg_tok_emb = TokenEmbedding(self.trg_vocab, config.model['dim_model'])
        self.positional_encoding = PositionalEncoding(config)

        self.device = torch.device(config.model['device'])
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor,
                src_padding_mask: Tensor,
                trg_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None,
                                src_padding_mask, trg_padding_mask, memory_key_padding_mask)
        return self.linear(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                                        self.src_tok_emb(src)), src_mask)

    def decode(self, trg: Tensor, memory: Tensor, trg_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                                        self.trg_tok_emb(trg)), memory,
                                        trg_mask)


def build_model(config: ModuleType):
    """
    encoder = Encoder(src_vocab_size=src_vocab,
                      dim_model=config.model['dim_model'],
                      num_layers=config.model['num_layers'],
                      num_heads=config.model['num_heads'],
                      dim_ff=config.model['dim_ff'],
                      dropout=config.model['dropout'],
                      max_seq_len=config.model['max_seq_len'],
                      padding_idx=0)
    decoder = Decoder(trg_vocab_size=trg_vocab,
                      dim_model=config.model['dim_model'],
                      num_layers=config.model['num_layers'],
                      num_heads=config.model['num_heads'],
                      dim_ff=config.model['dim_ff'],
                      dropout=config.model['dropout'],
                      max_seq_len=config.model['max_seq_len'],
                      padding_idx=0)
    head = ClassifyHead(dim_model=config.model['dim_model'],
                        trg_vocab_size=trg_vocab)

    model = Transformer(encoder=encoder, decoder=decoder, classifier=head).to(torch.device(config.model['device']))
    model.config = config
    """
    model = TransformerModel(config)
    print('Model successfully initialized')
    if config.model['load'] is not None:
        model.load_state_dict(torch.load(config.model['load']))
        print('Weights loaded successfully ')
    return model


def build_criterion(config: ModuleType) -> nn.Module:
    loss_class = getattr(nn, config.model['criterion']['name'], None)
    if loss_class is not None:
        print('Criterion successfully build')
        return loss_class(**config.model['criterion']['args'])
    else:
        raise RuntimeError('Error in criterion name!')


def build_optimizer(config: ModuleType,
                    model: nn.Module) -> optim:
    optimizer_class = getattr(optim, config.model['optimizer']['name'], None)
    if optimizer_class is not None:
        return optimizer_class(model.parameters(), **config.model['optimizer']['args'])
    else:
        raise RuntimeError('Error in optimizer name!')


def build_metric(config: ModuleType) -> Metric:
    metric_class = getattr(torchmetrics.text, config.model['val_config']['metric']['name'], None)
    if metric_class is not None:
        print('Metric successfully build')
        return metric_class(**config.model['val_config']['metric']['args'])
    else:
        raise RuntimeError('Error with metric name!')
