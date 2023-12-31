import argparse
import torch

from assembler import load_config
from assembler.model import build_model
from assembler.data import get_dataloaders
from assembler.train import create_mask
from torchsummaryX import summary
from thop import profile, clever_format


def parse_args():
    parser = argparse.ArgumentParser(description='Script for summary model')
    parser.add_argument('config',
                        type=str,
                        help='Path to model config')
    parser.add_argument('-b',
                        '--bytes',
                        type=int,
                        help='Single parameter size (in bytes)',
                        default=4
                        )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    train_dataloader, _, _ = get_dataloaders(config)
    model = build_model(config)

    device = torch.device(config.model['device'])

    src, trg = next(iter(train_dataloader))
    src, trg = src.to(device), trg.to(device)
    src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src.transpose(0, 1),
                                                                         trg.transpose(0, 1)[:-1, :], device=device)
    summary(model,
            src,
            trg=trg[:, :-1],
            src_mask=src_mask,
            trg_mask=trg_mask,
            src_padding_mask=src_padding_mask,
            trg_padding_mask=trg_padding_mask,
            memory_key_padding_mask=src_padding_mask)

    # Memory info
    total_memory = sum(p.numel() for p in model.parameters()) * args.bytes
    print(f'Total memory: {round(total_memory / (1024 ** 2))} MB')

    # Flops info
    with torch.no_grad():
        flops, params = profile(model, (src, trg[:, :-1], src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask))
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    print(f"FLOPS: {flops_formatted}")
    print(f"Params: {params_formatted}")


if __name__ == '__main__':
    main()
