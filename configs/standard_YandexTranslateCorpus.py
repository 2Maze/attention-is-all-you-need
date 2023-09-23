from assembler.data import simple_pipeline


reproducibility = {
    'seed': 42
}

vocab = {
    'corpus_src': 'data/YandexTranslateCorpus/root/corpus.en_ru.1m.ru',
    'corpus_trg': 'data/YandexTranslateCorpus/root/corpus.en_ru.1m.en',
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
    'split': {'train': 0.8, 'val': 0.2},
    'train_corpus_src': vocab['corpus_src'],
    'train_corpus_trg': vocab['corpus_trg'],
    'val_corpus_src': None,
    'val_corpus_trg': None,
    'test_corpus_src': None,
    'test_corpus_trg': None,
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
