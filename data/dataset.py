import torch
import torch.nn.functional as F
import linecache

from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable


class TranslateDataset(Dataset):
    def __init__(self,
                 corpus_src: str,
                 corpus_trg: str,
                 vocab_src: str,
                 vocab_trg: str,
                 pad_token: str,
                 start_token: str,
                 end_token: str,
                 max_seq_len: int,
                 preprocess: Callable[[str, str], tuple[list, list]]) -> None:

        self.corpus_src = corpus_src
        self.corpus_trg = corpus_trg
        self.vocab_src = torch.load(vocab_src)
        self.vocab_trg = torch.load(vocab_trg)
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.max_seq_len = max_seq_len

        self.preprocess = preprocess

    def __len__(self) -> int:
        with open(self.corpus_src) as file:
            return sum(chunk.count('\n')
                       for chunk in iter(lambda: file.read(1 << 13), ''))

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        assert idx >= 0, '`idx` argument must be >= 0'

        src_text = linecache.getline(self.corpus_src, idx + 1)
        trg_text = linecache.getline(self.corpus_trg, idx + 1)

        src_tokens, trg_tokens = self.preprocess(src_text, trg_text)
        src_ids = [self.vocab_src[self.start_token]] + [self.vocab_src[token] for token in src_tokens][
                                                        :self.max_seq_len - 2] + [self.vocab_src[self.end_token]]
        trg_ids = [self.vocab_trg[self.start_token]] + [self.vocab_trg[token] for token in trg_tokens][
                                                        :self.max_seq_len - 2] + [self.vocab_trg[self.end_token]]
        src = torch.tensor(src_ids, dtype=torch.int64)
        trg = torch.tensor(trg_ids, dtype=torch.int64)

        return src, trg
