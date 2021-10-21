import numpy as np
import torch
from torch.autograd import Variable

from fairseq.data import data_utils


class SwitchOut(object):
    def __init__(self, src_dict, tgt_dict, switch_tau, raml_tau) -> None:
        super().__init__()
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.src_vocab_size = src_dict.__len__()
        self.tgt_vocab_size = tgt_dict.__len__()

        self.bos_id = src_dict.bos_index
        self.eos_id = src_dict.eos_index
        self.pad_id = src_dict.pad_index

        self.switch_tau = switch_tau
        self.raml_tau = raml_tau

    def switchout(self, sents, tau=0.1):
        bsz, n_steps = sents.size()

        # we don't want the tau to be dynamic
        if self.switch_tau is None:
            self.switch_tau = tau
        # compute mask for sents without  bos/eos/pad
        mask = torch.eq(sents, self.bos_id) | torch.eq(sents, self.eos_id) | torch.eq(sents, self.pad_id)
        lengths = (1.0 - mask.float()).sum(dim=1)

        # sample the number of words to corrupt fr each sentence
        logits = torch.arange(n_steps)
        logits = logits.float().mul_(-1).unsqueeze(0).expand_as(sents).contiguous().masked_fill_(mask, -float("inf"))
        logits = Variable(logits)  # adding to computation graph node
        probs = torch.nn.functional.softmax(logits.mul_(self.switch_tau), dim=1)
        num_words = torch.distributions.Categorical(probs).sample()

        # sample the corrupted positions
        corrupt_pos = (
            num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)
        )
        # import ipdb

        # ipdb.set_trace()
        corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).bool()
        total_words = int(corrupt_pos.sum())

        # sample the corrupted values to add to sents
        corrupt_val = torch.LongTensor(total_words)
        # starts from 2 because pad_idx = 1, eos_idx = 2 in fairseq dict
        # we don't want to replace tokens with bos/eos/pad token
        corrupt_val = corrupt_val.random_(2, self.src_vocab_size)
        corrupts = torch.zeros(bsz, n_steps).long()
        corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
        sampled_sents = sents.add(Variable(corrupts)).remainder_(self.src_vocab_size)

        return sampled_sents
