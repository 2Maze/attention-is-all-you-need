import torch
import re

from torch import Tensor
from types import ModuleType
from torch.utils.data import DataLoader
from data.dataset import YandexTranslateCorpus
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab, vocab
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import word_tokenize


class WordIDMapper:
    def __init__(self, config: ModuleType) -> None:
        self.config = config
        self.ru_vocab = torch.load(config.vocab['path_ru'])
        self.en_vocab = torch.load(config.vocab['path_en'])
        self.ru_id2word = self.ru_vocab.get_itos()
        self.en_id2word = self.en_vocab.get_itos()

    def ruids2word(self, ids: Tensor):
        assert ids.dim() == 1 or ids.dim() == 2
        if ids.dim() == 1:
            tokens = []
            for ruid in ids:
                token = self.ru_id2word[ruid]
                if token not in [self.config.vocab['special_tokens']['padding']]:
                    tokens.append(token)
            return ' '.join(tokens)
        elif ids.dim() == 2:
            batch = []
            for sent_id in ids:
                tokens = []
                for ruid in sent_id:
                    token = self.ru_id2word[ruid]
                    if token not in [self.config.vocab['special_tokens']['padding']]:
                        tokens.append(token)
                batch.append(' '.join(tokens))
            return batch

    def enids2word(self, ids: Tensor):
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


def build_dataset(config: ModuleType) -> YandexTranslateCorpus:
    dataset = YandexTranslateCorpus(**config.dataset)
    return dataset


def split_dataset(config: ModuleType,
                  dataset: YandexTranslateCorpus
                  ) -> tuple[Subset[YandexTranslateCorpus], Subset[YandexTranslateCorpus]]:
    reproducibility_config = config.reproducibility
    dataset_split_config = config.dataset_split

    train_indices, val_indices = train_test_split(range(len(dataset)),
                                                  train_size=dataset_split_config['train'],
                                                  test_size=dataset_split_config['validation'],
                                                  random_state=reproducibility_config['seed'],
                                                  )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_dataloaders(config: ModuleType,
                      datasets: tuple[Subset[YandexTranslateCorpus], Subset[YandexTranslateCorpus]]) -> tuple[DataLoader, DataLoader]:
    dataloader_config = config.dataloader
    train_dataset, val_dataset = datasets
    return (DataLoader(train_dataset,
                       shuffle=True,
                       batch_size=dataloader_config['batch_size'],
                       num_workers=dataloader_config['num_workers']),
            DataLoader(val_dataset,
                       shuffle=False,
                       batch_size=dataloader_config['batch_size'],
                       num_workers=dataloader_config['num_workers']))


def get_datasets(config) -> tuple[Subset[YandexTranslateCorpus], Subset[YandexTranslateCorpus]]:
    return split_dataset(config, build_dataset(config))


def get_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    return build_dataloaders(config, get_datasets(config))


def clean_text_ru(text: str) -> str:
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()


def clean_text_eng(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text


def clean_text(text: str, language: str) -> str:
    if language == 'ru':
        output = clean_text_ru(text)
    elif language == 'en':
        output = clean_text_eng(text)
    else:
        raise RuntimeError(f'Language - {language} not in support languages!')
    return output


def tokenize_ru(text: str) -> list:
    return word_tokenize(text, language='russian')


def tokenize_eng(text: str) -> list:
    return word_tokenize(text, language='english')


def tokenize(text: str, language: str) -> list:
    if language == 'ru':
        output = tokenize_ru(text)
    elif language == 'en':
        output = tokenize_eng(text)
    else:
        raise RuntimeError(f'Language - {language} not in support languages!')
    return output


def simple_pipeline(ru_text: str,
                    en_text: str) -> tuple[list, list]:
    clean_ru = clean_text(ru_text, language='ru')
    clean_en = clean_text(en_text, language='en')

    tokens_ru = tokenize(clean_ru, language='ru')
    tokens_en = tokenize(clean_en, language='en')
    return tokens_ru, tokens_en


def save_vocab(vocabulary: Vocab,
               out: str) -> None:
    torch.save(vocabulary, out)


def build_vocab(config: ModuleType) -> None:
    counter_ru = Counter()
    counter_en = Counter()

    with open(config.vocab['corpus_ru'], 'r') as corpus_1, open(config.vocab['corpus_en'], 'r') as corpus_2:
        ru_lines = corpus_1.readlines()
        en_lines = corpus_2.readlines()
        assert len(ru_lines) == len(en_lines)
        for ru_text, en_text in tqdm(zip(ru_lines, en_lines), total=len(ru_lines)):
            ru_norm, en_norm = simple_pipeline(ru_text, en_text)
            counter_ru.update(ru_norm)
            counter_en.update(en_norm)

    if config.vocab['max_tokens_ru'] is not None:
        counter_ru = Counter({key: count for key, count in counter_ru.most_common()[:config.vocab['max_tokens_ru']]})

    if config.vocab['max_tokens_en'] is not None:
        counter_en = Counter({key: count for key, count in counter_en.most_common()[:config.vocab['max_tokens_en']]})

    PAD = config.vocab['special_tokens']['padding']
    UNK = config.vocab['special_tokens']['unknown']
    BOS = config.vocab['special_tokens']['start']
    EOS = config.vocab['special_tokens']['end']

    specials = [PAD, UNK, BOS, EOS]
    vocab_ru, vocab_en = (vocab(counter_ru, specials=specials, min_freq=config.vocab['min_freq_ru']),
                          vocab(counter_en, specials=specials, min_freq=config.vocab['min_freq_en']))

    vocab_ru.set_default_index(vocab_ru[UNK])
    vocab_en.set_default_index(vocab_en[UNK])

    print(f'Russian vocab consist of {len(vocab_ru)} words')
    print(f'English vocab consist of {len(vocab_en)} words')

    save_vocab(vocab_ru, config.vocab['path_ru'])
    save_vocab(vocab_en, config.vocab['path_en'])