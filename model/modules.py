import torch.nn.functional as F

from torch import nn
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self,
                 dim_model: int,
                 dim_ff: int,
                 dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()

        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_2(F.relu(self.w_1(x))))


class ClassifyHead(nn.Module):
    def __init__(self,
                 dim_model: int,
                 trg_vocab_size: int) -> None:
        super(ClassifyHead, self).__init__()
        self.w = nn.Linear(dim_model, trg_vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(self.w(x), dim=-1)