from assembler.data import simple_pipeline


reproducibility = {
    'seed': 42
}

vocab = {
    'train_corpus_en': 'data/train.en',
    'train_corpus_de': 'data/train.de',
    'min_freq_en': 2,
    'min_freq_de': 2,
    'max_tokens_en': 25000,
    'max_tokens_de': 25000,
    'special_tokens':
    {
        'padding': '<pad>',
        'unknown': '<unk>',
        'start': '<bos>',
        'end': '<eos>',
    },
    'path_en': 'data/en.vocab',
    'path_de': 'data/de.vocab',
}

dataset = {
    'train_corpus_en': 'data/train.en',
    'train_corpus_de': 'data/train.de',
    'val_corpus_en': 'data/val.en',
    'val_corpus_de': 'data/val.de',
    'test_corpus_en': 'data/test.en',
    'test_corpus_de': 'data/test.de',
    'vocab_en': vocab['path_en'],
    'vocab_de': vocab['path_de'],
    'translate_to': 'de',
    'max_seq_len': 60,
    'pad_token': vocab['special_tokens']['padding'],
    'start_token': vocab['special_tokens']['start'],
    'end_token': vocab['special_tokens']['end'],
    'preprocess': simple_pipeline,
}

dataset_split = {
    'train': 0.7,
    'validation': 0.3,
}

dataloader = {
    'batch_size': 64,
    'num_workers': 2}

model = {
    'dim_model': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dim_ff': 2048,
    'dropout': 0.1,
    'max_seq_len': dataset['max_seq_len'],
    'load': None,
    'device': 'cuda:0',

    'criterion': {'name': 'CrossEntropyLoss',
                  'args': {'ignore_index': 0}},

    'optimizer': {'name': 'Adam',
                  'args': {'lr': 0.0001,
                           'betas': (0.9, 0.98),
                           'eps': 1e-9}},

    'train_config': {'epochs': 20,
                     'clip_gradient': 1.0},

    'val_config': {'metric': {'name': 'BLEUScore',
                              'args': {}}}
}
