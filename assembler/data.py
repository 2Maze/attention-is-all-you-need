import torch
import re
import string
import spacy

from typing import Sequence, Union
from torch import Tensor
from types import ModuleType
from torch.utils.data import DataLoader
from data.dataset import Multi30k
from torchtext.vocab import Vocab, vocab
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


spacy_en = spacy.load('en_core_web_md')
spacy_de = spacy.load('de_core_news_md')


class WordIDMapper:
    def __init__(self, config: ModuleType) -> None:
        self.config = config
        self.en_vocab = torch.load(config.vocab['path_en'])
        self.de_vocab = torch.load(config.vocab['path_de'])
        self.en_id2word = self.en_vocab.get_itos()
        self.de_id2word = self.de_vocab.get_itos()
        self.translate_to = self.config.dataset['translate_to']

    def enids2words(self, ids: Tensor) -> Union[list, str]:
        assert ids.dim() == 1 or ids.dim() == 2
        if ids.dim() == 1:
            tokens = []
            for enid in ids:
                token = self.en_id2word[enid]
                if token not in [self.config.vocab['special_tokens']['padding']]:
                    tokens.append(token)
            return ' '.join(tokens)
        elif ids.dim() == 2:
            batch = []
            for sent_id in ids:
                tokens = []
                for enid in sent_id:
                    token = self.en_id2word[enid]
                    if token not in [self.config.vocab['special_tokens']['padding']]:
                        tokens.append(token)
                batch.append(' '.join(tokens))
            return batch

    def deids2words(self, ids: Tensor) -> Union[list, str]:
        assert ids.dim() == 1 or ids.dim() == 2
        if ids.dim() == 1:
            tokens = []
            for deid in ids:
                token = self.de_id2word[deid]
                if token not in [self.config.vocab['special_tokens']['padding']]:
                    tokens.append(token)
            return ' '.join(tokens)
        elif ids.dim() == 2:
            batch = []
            for sent_id in ids:
                tokens = []
                for deid in sent_id:
                    token = self.de_id2word[deid]
                    if token not in [self.config.vocab['special_tokens']['padding']]:
                        tokens.append(token)
                batch.append(' '.join(tokens))
            return batch

    def src2words(self, ids: Tensor) -> Union[list, str]:
        if self.translate_to == 'de':
            return self.enids2words(ids)
        elif self.translate_to == 'en':
            return self.deids2words(ids)
        else:
            raise RuntimeError('Error in translate_to')

    def trg2words(self, ids: Tensor) -> Union[list, str]:
        if self.translate_to == 'de':
            return self.deids2words(ids)
        elif self.translate_to == 'en':
            return self.enids2words(ids)
        else:
            raise RuntimeError('Error in translate_to')


def preprocessing_text(text):
    text = text.lower().strip()
    text = re.sub(f'[{string.punctuation}\n]', '', text)
    return text


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def simple_pipeline(en_text: str,
                    de_text: str) -> tuple[list, list]:
    clean_en = preprocessing_text(en_text)
    clean_de = preprocessing_text(de_text)

    tokens_en = tokenize_en(clean_en)
    tokens_de = tokenize_de(clean_de)
    return tokens_en, tokens_de


def save_vocab(vocabulary: Vocab,
               out: str) -> None:
    torch.save(vocabulary, out)


def build_vocab(config: ModuleType) -> None:
    counter_en = Counter()
    counter_de = Counter()

    with open(config.vocab['train_corpus_en'], 'r') as corpus_en, open(config.vocab['train_corpus_de'], 'r') as corpus_de:
        en_lines = corpus_en.readlines()
        de_lines = corpus_de.readlines()
        for en_text, de_text in tqdm(zip(en_lines, de_lines), total=len(en_lines)):
            en_tokens, de_tokens = simple_pipeline(en_text, de_text)
            counter_en.update(en_tokens)
            counter_de.update(de_tokens)

    if config.vocab['max_tokens_en'] is not None:
        counter_en = Counter({key: count for key, count in counter_en.most_common()[:config.vocab['max_tokens_en']]})

    if config.vocab['max_tokens_de'] is not None:
        counter_de = Counter({key: count for key, count in counter_de.most_common()[:config.vocab['max_tokens_de']]})

    PAD = config.vocab['special_tokens']['padding']
    UNK = config.vocab['special_tokens']['unknown']
    BOS = config.vocab['special_tokens']['start']
    EOS = config.vocab['special_tokens']['end']

    specials = [PAD, UNK, BOS, EOS]
    vocab_en, vocab_de = (vocab(counter_en, specials=specials, min_freq=config.vocab['min_freq_en']),
                          vocab(counter_de, specials=specials, min_freq=config.vocab['min_freq_de']))

    vocab_en.set_default_index(vocab_en[UNK])
    vocab_de.set_default_index(vocab_de[UNK])

    print(f'English vocab consist of {len(vocab_en)} words')
    print(f'German vocab consist of {len(vocab_de)} words')

    save_vocab(vocab_en, config.vocab['path_en'])
    save_vocab(vocab_de, config.vocab['path_de'])


def build_dataset(config: ModuleType) -> Sequence[Dataset]:
    train_dataset = Multi30k(corpus_en=config.dataset['train_corpus_en'],
                             corpus_de=config.dataset['train_corpus_de'],
                             vocab_en=config.dataset['vocab_en'],
                             vocab_de=config.dataset['vocab_de'],
                             translate_to=config.dataset['translate_to'],
                             pad_token=config.dataset['pad_token'],
                             start_token=config.dataset['start_token'],
                             end_token=config.dataset['end_token'],
                             max_seq_len=config.dataset['max_seq_len'],
                             preprocess=config.dataset['preprocess'])
    val_dataset = Multi30k(corpus_en=config.dataset['val_corpus_en'],
                           corpus_de=config.dataset['val_corpus_de'],
                           vocab_en=config.dataset['vocab_en'],
                           vocab_de=config.dataset['vocab_de'],
                           translate_to=config.dataset['translate_to'],
                           pad_token=config.dataset['pad_token'],
                           start_token=config.dataset['start_token'],
                           end_token=config.dataset['end_token'],
                           max_seq_len=config.dataset['max_seq_len'],
                           preprocess=config.dataset['preprocess'])
    test_dataset = Multi30k(corpus_en=config.dataset['test_corpus_en'],
                            corpus_de=config.dataset['test_corpus_de'],
                            vocab_en=config.dataset['vocab_en'],
                            vocab_de=config.dataset['vocab_de'],
                            translate_to=config.dataset['translate_to'],
                            pad_token=config.dataset['pad_token'],
                            start_token=config.dataset['start_token'],
                            end_token=config.dataset['end_token'],
                            max_seq_len=config.dataset['max_seq_len'],
                            preprocess=config.dataset['preprocess'])
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(config: ModuleType,
                      datasets: Sequence[Dataset]) -> Sequence[DataLoader]:
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(src_sample)
            trg_batch.append(trg_sample)

        return (pad_sequence(src_batch, padding_value=0, batch_first=True),
                pad_sequence(trg_batch, padding_value=0, batch_first=True))
    is_train = True
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset,
                                      shuffle=True if is_train else False,
                                      batch_size=config.dataloader['batch_size'],
                                      num_workers=config.dataloader['num_workers'],
                                      collate_fn=collate_fn))
        is_train = False
    return dataloaders


def get_datasets(config: ModuleType) -> Sequence[Dataset]:
    return build_dataset(config)


def get_dataloaders(config: ModuleType) -> Sequence[DataLoader]:
    return build_dataloaders(config, build_dataset(config))


def get_mapper(config: ModuleType) -> WordIDMapper:
    return WordIDMapper(config)
