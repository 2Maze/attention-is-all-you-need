import argparse

from assembler import load_config
from assembler.data import get_mapper
from assembler.model import build_model
from assembler.inference import inference_model


def parse_args():
    parser = argparse.ArgumentParser(description='Script for train model')
    parser.add_argument('load',
                        type=str,
                        help='Path to model weights')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='Path to model config',
                        default='configs/standard.py')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config.model['load'] = args.load
    mapper = get_mapper(config)
    model = build_model(config)
    model.eval()
    model_input_text = input(f'Write something on {"german" if config.dataset["translate_to"] == "en" else "english"}: ')
    while model_input_text != '':
        translated_ids = inference_model(model_input_text=model_input_text,
                                         config=config,
                                         mapper=mapper,
                                         model=model)
        print(
            f'Model translate: {mapper.trg2words(translated_ids).replace(config.vocab["special_tokens"]["start"], "").replace(config.vocab["special_tokens"]["end"], "")}')
        model_input_text = input(
            f'Write something on {"german" if config.dataset["translate_to"] == "en" else "english"}: ')


if __name__ == '__main__':
    main()