import argparse

from assembler import load_config
from assembler.data import get_dataloaders, get_mapper
from assembler.model import build_model, build_criterion, build_optimizer, build_metric
from assembler.train import start_train


def parse_args():
    parser = argparse.ArgumentParser(description='Script for train model')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='Path to model config',
                        default='configs/standard.py',)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config)
    mapper = get_mapper(config)
    model = build_model(config)
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    metric = build_metric(config)
    print(len(train_dataloader), len(val_dataloader))
    start_train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        mapper=mapper,
        optimizer=optimizer,
        metric=metric,
        **config.model['train_config'])


if __name__ == '__main__':
    main()
