import argparse

from assembler import load_config
from assembler.data import get_dataloaders, WordIDMapper


def parse_args():
    parser = argparse.ArgumentParser(description='Script for browse dataset vocabs')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='Path to model config',
                        default='configs/standard.py')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        help='Batch size of output',
                        default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config.dataloader['batch_size'] = args.batch_size
    train_dataloader, val_dataloader = get_dataloaders(config)
    mapper = WordIDMapper(config)
    ru, en = next(iter(train_dataloader))
    print('_______________________________')
    for i in range(len(ru)):
        print(f'ru {"(trg)" if "ru" == config.dataset["translate_to"] else "(src)"}: ', mapper.ruids2word(ru[i]))
        print(f'en {"(trg)" if "en" == config.dataset["translate_to"] else "(src)"}: ', mapper.enids2word(en[i]))
        print('_______________________________')


if __name__ == '__main__':
    main()
