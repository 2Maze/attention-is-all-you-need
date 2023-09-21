import argparse

from assembler import load_config
from assembler.data import get_dataloaders, get_mapper


def parse_args():
    parser = argparse.ArgumentParser(description='Script for browse dataset vocabs')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='Path to model config',
                        default='configs/standard.py')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config)
    mapper = get_mapper(config)
    src, trg = next(iter(val_dataloader))

    print('We have train dataset: {}, val dataset: {}, test_dataset: {}'.format(train_dataloader is not None,
                                                                                val_dataloader is not None,
                                                                                test_dataloader is not None))

    print('_________________________')
    for s, t in zip(mapper.src_ids2words(src), mapper.trg_ids2words(trg)):
        print('src: {}'.format(s))
        print('trg: {}'.format(t))
        print('_________________________')
    print('src batch shape {} \ntrg batch shape {}'.format(src.shape, trg.shape))


if __name__ == '__main__':
    main()
