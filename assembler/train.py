import torch
import os
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from torchtext.data import metrics
from torch.utils.tensorboard import SummaryWriter
from assembler.data import WordIDMapper


def train_iter(epoch: int,
               model: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               optimizer: optim,
               logger: SummaryWriter,
               clip_gradient: float) -> None:
    model.train()
    device = torch.device(model.config.model['device'])

    for n_iter, batch in enumerate(tqdm(dataloader)):
        n_iter += epoch * len(dataloader)
        ru, en = batch
        ru, en = ru.to(device), en.to(device)

        if model.config.dataset['translate_to'] == 'ru':
            src = en
            trg = ru
        elif model.config.dataset['translate_to'] == 'en':
            src = ru
            trg = en
        else:
            raise RuntimeError('Error in translate_to')

        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        loss = criterion(
            output.view(-1, output.size(-1)),  # (batch_size * (target_seq_len - 1), vocab_size)
            trg[:, 1:].contiguous().view(-1)  # (batch_size * (target_seq_len - 1))
        )
        # Save iter loss
        logger.add_scalar('Loss/train', loss.item(), n_iter)

        # Backward
        loss.backward()
        # Clip gradient
        # nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
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
    device = torch.device(model.config.model['device'])

    for batch in tqdm(dataloader):
        ru, en = batch
        ru, en = ru.to(device), en.to(device)

        if model.config.dataset['translate_to'] == 'ru':
            src = en
            trg = ru
        elif model.config.dataset['translate_to'] == 'en':
            src = ru
            trg = en
        else:
            raise RuntimeError('Error in translate_to')
        output, _ = model(src, trg[:, :-1])
        loss = criterion(
            output.view(-1, output.size(-1)),  # (batch_size * (target_seq_len - 1), vocab_size)
            trg[:, 1:].contiguous().view(-1)  # (batch_size * (target_seq_len - 1))
        )

        # Compute total loss
        total_loss += loss.item()

        # Compute metric
        output = output.argmax(dim=-1)  # (batch_size, target_seq_len - 1)

        if model.config.dataset['translate_to'] == 'ru':
            pred_tokens_batch = mapper.ruids2word(output)
            trg_tokens_batch = [[sent] for sent in mapper.ruids2word(trg[:, 1:])]
        elif model.config.dataset['translate_to'] == 'en':
            pred_tokens_batch = mapper.enids2word(output)
            trg_tokens_batch = [[sent] for sent in mapper.enids2word(trg[:, 1:])]
        else:
            raise RuntimeError('Error in translate_to')

        bleu = metric(pred_tokens_batch, trg_tokens_batch)
        total_bleu += bleu

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

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        train_iter(epoch=epoch,
                   model=model,
                   dataloader=train_dataloader,
                   criterion=criterion,
                   optimizer=optimizer,
                   logger=writer,
                   clip_gradient=clip_gradient)

        val_iter(epoch=epoch,
                 model=model,
                 dataloader=val_dataloader,
                 criterion=criterion,
                 mapper=mapper,
                 logger=writer,
                 metric=metric)

        save_model(epoch=epoch,
                   model=model)
