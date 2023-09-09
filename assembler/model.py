import torch
import torchmetrics
import torch.nn as nn

from torchmetrics import Metric
from torch import optim
from types import ModuleType
from model.encoder import Encoder
from model.decoder import Decoder
from model.modules import ClassifyHead
from model.transformer import Transformer


def build_model(config: ModuleType):
    vocab_ru = torch.load(config.vocab['path_ru'])
    vocab_en = torch.load(config.vocab['path_en'])

    encoder = Encoder(src_vocab_size=len(vocab_ru) if config.dataset['translate_to'] == 'en' else len(vocab_en),
                      dim_model=config.model['dim_model'],
                      num_layers=config.model['num_layers'],
                      num_heads=config.model['num_heads'],
                      dim_ff=config.model['dim_ff'],
                      dropout=config.model['dropout'],
                      max_seq_len=config.model['max_seq_len'],
                      padding_idx=0)
    decoder = Decoder(trg_vocab_size=len(vocab_en) if config.dataset['translate_to'] == 'en' else len(vocab_ru),
                      dim_model=config.model['dim_model'],
                      num_layers=config.model['num_layers'],
                      num_heads=config.model['num_heads'],
                      dim_ff=config.model['dim_ff'],
                      dropout=config.model['dropout'],
                      max_seq_len=config.model['max_seq_len'],
                      padding_idx=0)
    head = ClassifyHead(dim_model=config.model['dim_model'],
                        trg_vocab_size=len(vocab_en) if config.dataset['translate_to'] == 'en' else len(vocab_ru))

    model = Transformer(encoder=encoder, decoder=decoder, classifier=head).to(torch.device(config.model['device']))
    model.config = config
    print('Model successfully initialized')
    if config.model['load'] is not None:
        model.load_state_dict(torch.load(config.model['load']))
        print('Weights loaded successfully ')
    return model


def build_criterion(config: ModuleType) -> nn.Module:
    loss_class = getattr(nn, config.model['criterion']['name'], None)
    if loss_class is not None:
        print('Criterion successfully build')
        return loss_class(**config.model['criterion']['params'])
    else:
        raise RuntimeError('Error in criterion name!')


def build_optimizer(config: ModuleType,
                    model: nn.Module) -> optim:
    optimizer_class = getattr(optim, config.model['optimizer']['name'], None)
    if optimizer_class is not None:
        print('Optimizer successfully build')
        return optimizer_class(model.parameters(), **config.model['optimizer']['params'])
    else:
        raise RuntimeError('Error in optimizer name!')


def build_metric(config: ModuleType) -> Metric:
    metric_class = getattr(torchmetrics.text, config.model['val_config']['metric']['name'], None)
    if metric_class is not None:
        print('Metric successfully build')
        return metric_class(**config.model['val_config']['metric']['params'])
    else:
        raise RuntimeError('Error with metric name!')
