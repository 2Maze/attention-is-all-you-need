import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from torchtext.data import metrics
from torch.utils.tensorboard import SummaryWriter
from assembler.data import WordIDMapper


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask.to(device), tgt_mask, src_padding_mask.to(device), tgt_padding_mask.to(device)


def train_iter(epoch: int,
               model: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               optimizer: optim,
               logger: SummaryWriter,
               clip_gradient: float) -> None:
    model.train()
    device = model.device

    for n_iter, batch in enumerate(tqdm(dataloader)):
        n_iter += epoch * len(dataloader)
        src, trg = batch
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src.transpose(0, 1),
                                                                             trg.transpose(0, 1)[:-1, :], device=device)

        logits = model(src=src,
                       trg=trg[:, :-1],
                       src_mask=src_mask,
                       trg_mask=trg_mask,
                       src_padding_mask=src_padding_mask,
                       trg_padding_mask=trg_padding_mask,
                       memory_key_padding_mask=src_padding_mask)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), trg[:, 1:].reshape(-1))
        loss.backward()

        # Save iter loss
        logger.add_scalar('Loss/train', loss.item(), n_iter)

        # Backward
        # Clip gradient
        nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        # Optimizer step
        optimizer.step()


@torch.no_grad()
def val_iter(epoch: int,
             model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             mapper: WordIDMapper,
             logger: SummaryWriter,
             metric: metrics):
    model.eval()
    total_loss = 0.
    total_bleu = 0.
    device = model.device

    for batch in tqdm(dataloader):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src.transpose(0, 1),
                                                                             trg.transpose(0, 1)[:-1, :], device=device)
        logits = model(src=src,
                       trg=trg[:, :-1],
                       src_mask=src_mask,
                       trg_mask=trg_mask,
                       src_padding_mask=src_padding_mask,
                       trg_padding_mask=trg_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        probs = F.softmax(logits, dim=-1)
        total_loss += criterion(logits.reshape(-1, logits.shape[-1]), trg[:, 1:].reshape(-1))
        bleu = metric(mapper.trg_ids2words(probs.argmax(dim=-1)), [[sent] for sent in mapper.trg_ids2words(trg)])
        total_bleu += bleu

    logger.add_text('Text/val', mapper.trg_ids2words(probs.argmax(dim=-1)[0]), epoch)
    logger.add_scalar('Loss/val', total_loss / len(dataloader), epoch)
    logger.add_scalar('BLEU/val', total_bleu / len(dataloader), epoch)


def save_model(epoch: int,
               model: nn.Module,
               path: str = 'weights') -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f'epoch_{epoch}.pth'))
    print(f'Save {epoch} epoch')


def start_train(model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                criterion: nn.Module,
                mapper: WordIDMapper,
                optimizer: optim,
                metric: metrics,
                epochs: int,
                clip_gradient: float) -> None:
    writer = SummaryWriter()

    assert train_dataloader is not None

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        train_iter(epoch=epoch,
                   model=model,
                   dataloader=train_dataloader,
                   criterion=criterion,
                   optimizer=optimizer,
                   logger=writer,
                   clip_gradient=clip_gradient)
        if val_dataloader is not None:
            val_iter(epoch=epoch,
                     model=model,
                     dataloader=val_dataloader,
                     criterion=criterion,
                     mapper=mapper,
                     logger=writer,
                     metric=metric)
        save_model(epoch=epoch,
                   model=model)
