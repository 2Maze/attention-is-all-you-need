import torch
import torch.nn as nn
import torch.nn.functional as F

from types import ModuleType
from assembler.data import WordIDMapper
from assembler.train import generate_square_subsequent_mask


def inference_model(model_input_text: str,
                    config: ModuleType,
                    mapper: WordIDMapper,
                    model: nn.Module):
    if config.dataset["translate_to"] == "en":
        _, src_tokens = config.dataset['preprocess'](en_text='', de_text=model_input_text)
        ids = [mapper.de_vocab[config.vocab['special_tokens']['start']]]
        for token in src_tokens:
            ids.append(mapper.de_vocab[token])
        ids.append(mapper.de_vocab[config.vocab['special_tokens']['end']])
    elif config.dataset["translate_to"] == "de":
        src_tokens, _ = config.dataset['preprocess'](en_text=model_input_text, de_text='')
        ids = [mapper.en_vocab[config.vocab['special_tokens']['start']]]
        for token in src_tokens:
            ids.append(mapper.en_vocab[token])
        ids.append(mapper.en_vocab[config.vocab['special_tokens']['end']])
    else:
        raise RuntimeError('Error in translate_to!')
    src = torch.tensor([ids]).to(model.device)
    src_mask = (torch.zeros(len(ids), len(ids))).type(torch.bool).to(model.device)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[2]]).type(torch.long).to(model.device)
    for i in range(config.dataset['max_seq_len'] - 1):
        trg_mask = (generate_square_subsequent_mask(len(ys[0]), device=model.device)
                    .type(torch.bool)).to(model.device)
        out = model.decode(ys, memory, trg_mask)
        prob = F.softmax(model.linear(out[:, -1]), dim=-1)
        next_word_id = prob.argmax(dim=-1)
        ys = torch.cat([ys, torch.tensor([[next_word_id]], device=model.device)], dim=-1)
        if next_word_id.item() == 3:
            break
    return ys[0]