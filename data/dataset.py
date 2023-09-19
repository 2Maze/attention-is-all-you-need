import torch
import torch.nn.functional as F
import linecache

from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable


class Multi30k(Dataset):
    def __init__(self,
                 corpus_en: str,
                 corpus_de: str,
                 vocab_en: str,
                 vocab_de: str,
                 translate_to: str,
                 pad_token: str,
                 start_token: str,
                 end_token: str,
                 max_seq_len: int,
                 preprocess: Callable[[str, str], tuple[list, list]]) -> None:
        assert translate_to in ['de', 'en']

        self.corpus_en = corpus_en
        self.corpus_de = corpus_de
        self.vocab_en = torch.load(vocab_en)
        self.vocab_de = torch.load(vocab_de)
        self.translate_to = translate_to
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.max_seq_len = max_seq_len

        self.preprocess = preprocess

    def __len__(self) -> int:
        with open(self.corpus_en) as file:
            return sum(chunk.count('\n')
                       for chunk in iter(lambda: file.read(1 << 13), ''))

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        assert idx >= 0, '`idx` argument must be >= 0'

        en_text = linecache.getline(self.corpus_en, idx + 1)
        de_text = linecache.getline(self.corpus_de, idx + 1)

        ru_tokens, en_tokens = self.preprocess(en_text, de_text)
        en_ids = [self.vocab_en[self.start_token]] + [self.vocab_en[token] for token in ru_tokens][
                                                     :self.max_seq_len - 2] + [self.vocab_en[self.end_token]]
        de_ids = [self.vocab_de[self.start_token]] + [self.vocab_de[token] for token in en_tokens][
                                                     :self.max_seq_len - 2] + [self.vocab_de[self.end_token]]
        if self.translate_to == 'de':
            src = torch.tensor(en_ids, dtype=torch.int64)
            trg = torch.tensor(de_ids, dtype=torch.int64)
        elif self.translate_to == 'en':
            src = torch.tensor(de_ids, dtype=torch.int64)
            trg = torch.tensor(en_ids, dtype=torch.int64)
        else:
            raise RuntimeError('Error in translate_to')
        return src, trg
