from assembler.data import simple_pipeline


reproducibility = {
    'seed': 42
}

vocab = {
    'train_corpus_src': 'data/Multi30k/train.en',
    'train_corpus_trg': 'data/Multi30k/train.de',
    'min_freq_src': 2,
    'min_freq_trg': 2,
    'max_tokens_src': 25000,
    'max_tokens_trg': 25000,
    'special_tokens':
    {
        'padding': '<pad>',
        'unknown': '<unk>',
        'start': '<bos>',
        'end': '<eos>',
    },
    'path_src': 'data/en.vocab',
    'path_trg': 'data/de.vocab',
}

dataset = {
    'train_corpus_src': vocab['train_corpus_src'],
    'train_corpus_trg': vocab['train_corpus_trg'],
    'val_corpus_src': 'data/Multi30k/val.en',
    'val_corpus_trg': 'data/Multi30k/val.de',
    'test_corpus_src': 'data/Multi30k/test.en',
    'test_corpus_trg': 'data/Multi30k/test.de',
    'vocab_src': vocab['path_src'],
    'vocab_trg': vocab['path_trg'],
    'max_seq_len': 60,
    'pad_token': vocab['special_tokens']['padding'],
    'start_token': vocab['special_tokens']['start'],
    'end_token': vocab['special_tokens']['end'],
    'preprocess': simple_pipeline,
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
