import argparse

from assembler import load_config
from assembler.data import build_vocab


def parse_args():
    parser = argparse.ArgumentParser(description='Script for creating vocabs')
    parser.add_argument('config',
                        type=str,
                        help='Path to model config')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    build_vocab(config)


if __name__ == '__main__':
    main()