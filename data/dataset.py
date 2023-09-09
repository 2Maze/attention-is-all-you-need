import linecache
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable


class YandexTranslateCorpus(Dataset):
    def __init__(self,
                 corpus_ru: str,
                 corpus_en: str,
                 vocab_ru: str,
                 vocab_en: str,
                 translate_to: str = 'en',
                 max_seq_len: int = 60,
                 preprocess: Callable[[str, str], tuple[list, list]] = lambda ru, en: (
                         ru.lower().split(), en.lower().split()),
                 pad_token: str = '<pad>',
                 start_token: str = '<bos>',
                 end_token: str = '<eos>') -> None:
        assert translate_to in ['en', 'ru']

        self.corpus_ru = corpus_ru
        self.corpus_en = corpus_en
        self.vocab_ru = torch.load(vocab_ru)
        self.vocab_en = torch.load(vocab_en)
        self.translate_to = translate_to
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

        self.preprocess = preprocess

    def __len__(self) -> int:
        with open(self.corpus_ru) as file:
            return sum(chunk.count('\n')
                       for chunk in iter(lambda: file.read(1 << 13), ''))

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        assert idx >= 0, '`idx` argument must be >= 0'

        ru_text = linecache.getline(self.corpus_ru, idx + 1)
        en_text = linecache.getline(self.corpus_en, idx + 1)

        ru_tokens, en_tokens = self.preprocess(ru_text, en_text)
        if self.translate_to == 'en':
            ru_ids = [self.vocab_ru[self.start_token]] + [self.vocab_ru[token] for token in ru_tokens][:self.max_seq_len - 1]
            en_ids = [self.vocab_en[self.start_token]] + [self.vocab_en[token] for token in en_tokens][:self.max_seq_len - 2] + [self.vocab_en[self.end_token]]
        elif self.translate_to == 'ru':
            ru_ids = [self.vocab_ru[self.start_token]] + [self.vocab_ru[token] for token in ru_tokens][:self.max_seq_len - 2] + [self.vocab_ru[self.end_token]]
            en_ids = [self.vocab_en[self.start_token]] + [self.vocab_en[token] for token in en_tokens][:self.max_seq_len - 1]
        else:
            raise RuntimeError()

        ru_ids = torch.tensor(ru_ids, dtype=torch.int64)
        en_ids = torch.tensor(en_ids, dtype=torch.int64)
        return (F.pad(ru_ids, (0, self.max_seq_len - len(ru_ids)), 'constant', 0),
                F.pad(en_ids, (0, self.max_seq_len - len(en_ids)), 'constant', 0))
