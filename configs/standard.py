reproducibility = {
    'seed': 42
}

vocab = {
    'corpus_ru': 'data/corpus.en_ru.1m.ru',
    'corpus_en': 'data/corpus.en_ru.1m.en',
    'min_freq_ru': 5,
    'min_freq_en': 5,
    'max_tokens_ru': 25000,
    'max_tokens_en': 25000,
    'special_tokens':
        {
            'padding': '<pad>',
            'unknown': '<unk>',
            'start': '<bos>',
            'end': '<eos>',
        },
    'path_ru': 'data/ru.pth',
    'path_en': 'data/en.pth',
}

dataset = {
    'corpus_ru': vocab['corpus_ru'],
    'corpus_en': vocab['corpus_en'],
    'vocab_ru': vocab['path_ru'],
    'vocab_en': vocab['path_en'],
    'translate_to': 'ru',
    'max_seq_len': 60,
    'pad_token': vocab['special_tokens']['padding'],
    'start_token': vocab['special_tokens']['start'],
    'end_token': vocab['special_tokens']['end'],
}

dataset_split = {
    'train': 0.7,
    'validation': 0.3,
}

dataloader = {
    'batch_size': 32,
    'num_workers': 2}

model = {
    'dim_model': 128,
    'num_layers': 3,
    'num_heads': 4,
    'dim_ff': 2048,
    'dropout': 0.1,
    'max_seq_len': dataset['max_seq_len'],
    'load': None,
    'device': 'cuda:0',

    'criterion': {'name': 'CrossEntropyLoss',
                  'params': {'ignore_index': 0}},

    'optimizer': {'name': 'Adam',
                  'params': {'lr': 1e-3,
                             'betas': (0.9, 0.98),
                             'eps': 1e-9}},

    'train_config': {'epochs': 15,
                     'clip_gradient': 1.0},

    'val_config': {'metric': {'name': 'BLEUScore',
                              'params': {}}}
}
