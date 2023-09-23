import os.path

import torch
import re
import string
import spacy

from typing import Sequence, Union
from torch import Tensor
from types import ModuleType
from torch.utils.data import DataLoader, Subset
from data.dataset import TranslateDataset
from torchtext.vocab import Vocab, vocab
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

spacy_en = spacy.load('en_core_web_md')
spacy_de = spacy.load('de_core_news_md')


class WordIDMapper:
    def __init__(self, config: ModuleType) -> None:
        self.config = config
        self.src_vocab = torch.load(config.vocab['path_src'])
        self.trg_vocab = torch.load(config.vocab['path_trg'])
        self.src_id2word = self.src_vocab.get_itos()
        self.trg_id2word = self.trg_vocab.get_itos()

    def src_ids2words(self, ids: Tensor) -> Union[list, str]:
        assert ids.dim() == 1 or ids.dim() == 2
        if ids.dim() == 1:
            tokens = []
            for src_id in ids:
                token = self.src_id2word[src_id]
                if token not in [self.config.vocab['special_tokens']['padding']]:
                    tokens.append(token)
            return ' '.join(tokens)
        elif ids.dim() == 2:
            batch = []
            for sent_id in ids:
                tokens = []
                for src_id in sent_id:
                    token = self.src_id2word[src_id]
                    if token not in [self.config.vocab['special_tokens']['padding']]:
                        tokens.append(token)
                batch.append(' '.join(tokens))
            return batch

    def trg_ids2words(self, ids: Tensor) -> Union[list, str]:
        assert ids.dim() == 1 or ids.dim() == 2
        if ids.dim() == 1:
            tokens = []
            for trg_id in ids:
                token = self.trg_id2word[trg_id]
                if token not in [self.config.vocab['special_tokens']['padding']]:
                    tokens.append(token)
            return ' '.join(tokens)
        elif ids.dim() == 2:
            batch = []
            for sent_id in ids:
                tokens = []
                for trg_id in sent_id:
                    token = self.trg_id2word[trg_id]
                    if token not in [self.config.vocab['special_tokens']['padding']]:
                        tokens.append(token)
                batch.append(' '.join(tokens))
            return batch


def preprocessing_text(text):
    text = text.lower().strip()
    text = re.sub(f'[{string.punctuation}\n]', '', text)
    return text


def tokenize_src(text):
    return text.split(' ')


def tokenize_trg(text):
    return text.split(' ')


def simple_pipeline(src_text: str,
                    trg_text: str) -> tuple[list, list]:
    clean_en = preprocessing_text(src_text)
    clean_de = preprocessing_text(trg_text)

    tokens_en = tokenize_src(clean_en)
    tokens_de = tokenize_trg(clean_de)
    return tokens_en, tokens_de


def save_vocab(vocabulary: Vocab,
               out: str) -> None:
    torch.save(vocabulary, out)


def build_vocab(config: ModuleType) -> None:
    counter_src = Counter()
    counter_trg = Counter()

    with open(config.vocab['corpus_src'], 'r') as corpus_en, open(config.vocab['corpus_trg'],
                                                                  'r') as corpus_de:
        src_lines = corpus_en.readlines()
        trg_lines = corpus_de.readlines()
        for src_text, trg_text in tqdm(zip(src_lines, trg_lines), total=len(src_lines)):
            src_tokens, trg_tokens = simple_pipeline(src_text, trg_text)
            counter_src.update(src_tokens)
            counter_trg.update(trg_tokens)

    if config.vocab['max_tokens_src'] is not None:
        counter_src = Counter({key: count for key, count in counter_src.most_common()[:config.vocab['max_tokens_src']]})

    if config.vocab['max_tokens_trg'] is not None:
        counter_trg = Counter({key: count for key, count in counter_trg.most_common()[:config.vocab['max_tokens_trg']]})

    PAD = config.vocab['special_tokens']['padding']
    UNK = config.vocab['special_tokens']['unknown']
    BOS = config.vocab['special_tokens']['start']
    EOS = config.vocab['special_tokens']['end']

    specials = [PAD, UNK, BOS, EOS]
    vocab_src, vocab_trg = (vocab(counter_src, specials=specials, min_freq=config.vocab['min_freq_src']),
                            vocab(counter_trg, specials=specials, min_freq=config.vocab['min_freq_trg']))

    vocab_src.set_default_index(vocab_src[UNK])
    vocab_trg.set_default_index(vocab_trg[UNK])

    print(f'Source vocab consist of {len(vocab_src)} words')
    print(f'Target vocab consist of {len(vocab_trg)} words')

    save_vocab(vocab_src, config.vocab['path_src'])
    save_vocab(vocab_trg, config.vocab['path_trg'])


def build_dataset(config: ModuleType) -> Sequence[Dataset]:
    if 'split' not in config.dataset:
        have_train = 'train_corpus_src' in config.dataset and 'train_corpus_trg' in config.dataset and os.path.exists(
            config.dataset['train_corpus_src']) and os.path.exists(config.dataset['train_corpus_trg'])
        have_val = 'val_corpus_src' in config.dataset and 'val_corpus_trg' in config.dataset and os.path.exists(
            config.dataset['val_corpus_src']) and os.path.exists(config.dataset['val_corpus_trg'])
        have_test = 'test_corpus_src' in config.dataset and 'test_corpus_trg' in config.dataset and os.path.exists(
            config.dataset['test_corpus_src']) and os.path.exists(config.dataset['test_corpus_trg'])

        if have_train:
            train_dataset = TranslateDataset(corpus_src=config.dataset['train_corpus_src'],
                                             corpus_trg=config.dataset['train_corpus_trg'],
                                             vocab_src=config.dataset['vocab_src'],
                                             vocab_trg=config.dataset['vocab_trg'],
                                             pad_token=config.dataset['pad_token'],
                                             start_token=config.dataset['start_token'],
                                             end_token=config.dataset['end_token'],
                                             max_seq_len=config.dataset['max_seq_len'],
                                             preprocess=config.dataset['preprocess'])
        else:
            train_dataset = None

        if have_val:
            val_dataset = TranslateDataset(corpus_src=config.dataset['val_corpus_src'],
                                           corpus_trg=config.dataset['val_corpus_trg'],
                                           vocab_src=config.dataset['vocab_src'],
                                           vocab_trg=config.dataset['vocab_trg'],
                                           pad_token=config.dataset['pad_token'],
                                           start_token=config.dataset['start_token'],
                                           end_token=config.dataset['end_token'],
                                           max_seq_len=config.dataset['max_seq_len'],
                                           preprocess=config.dataset['preprocess'])
        else:
            val_dataset = None

        if have_test:
            test_dataset = TranslateDataset(corpus_src=config.dataset['test_corpus_src'],
                                            corpus_trg=config.dataset['test_corpus_trg'],
                                            vocab_src=config.dataset['vocab_src'],
                                            vocab_trg=config.dataset['vocab_trg'],
                                            pad_token=config.dataset['pad_token'],
                                            start_token=config.dataset['start_token'],
                                            end_token=config.dataset['end_token'],
                                            max_seq_len=config.dataset['max_seq_len'],
                                            preprocess=config.dataset['preprocess'])
        else:
            test_dataset = None

        return train_dataset, val_dataset, test_dataset
    else:
        split = config.dataset['split']
        assert split['train'] + split['val'] == 1.
        dataset = TranslateDataset(corpus_src=config.dataset['train_corpus_src'],
                                   corpus_trg=config.dataset['train_corpus_trg'],
                                   vocab_src=config.dataset['vocab_src'],
                                   vocab_trg=config.dataset['vocab_trg'],
                                   pad_token=config.dataset['pad_token'],
                                   start_token=config.dataset['start_token'],
                                   end_token=config.dataset['end_token'],
                                   max_seq_len=config.dataset['max_seq_len'],
                                   preprocess=config.dataset['preprocess'])
        train_indices, val_indices = train_test_split(range(len(dataset)),
                                                      train_size=split['train'],
                                                      test_size=split['val'],
                                                      random_state=config.reproducibility['seed'],
                                                      )
        return Subset(dataset, train_indices), Subset(dataset, val_indices), None


def build_dataloaders(config: ModuleType,
                      datasets: Sequence[Dataset]) -> Sequence[DataLoader]:
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(src_sample)
            trg_batch.append(trg_sample)
        return (pad_sequence(src_batch, padding_value=0, batch_first=True),
                pad_sequence(trg_batch, padding_value=0, batch_first=True))

    if datasets[0] is not None:
        train_dataloader = DataLoader(datasets[0],
                                      shuffle=True,
                                      batch_size=config.dataloader['batch_size'],
                                      num_workers=config.dataloader['num_workers'],
                                      collate_fn=collate_fn)
    else:
        train_dataloader = None

    if datasets[1] is not None:
        val_dataloader = DataLoader(datasets[1],
                                    shuffle=True,
                                    batch_size=config.dataloader['batch_size'],
                                    num_workers=config.dataloader['num_workers'],
                                    collate_fn=collate_fn)
    else:
        val_dataloader = None

    if datasets[2] is not None:
        test_dataloader = DataLoader(datasets[2],
                                     shuffle=True,
                                     batch_size=config.dataloader['batch_size'],
                                     num_workers=config.dataloader['num_workers'],
                                     collate_fn=collate_fn)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader


def get_datasets(config: ModuleType) -> Sequence[Dataset]:
    return build_dataset(config)


def get_dataloaders(config: ModuleType) -> Sequence[DataLoader]:
    return build_dataloaders(config, build_dataset(config))


def get_mapper(config: ModuleType) -> WordIDMapper:
    return WordIDMapper(config)
