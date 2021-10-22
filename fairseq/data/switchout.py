import logging
import numpy as np
import torch
from torch.autograd import Variable

from fairseq.data import data_utils

logger = logging.getLogger(__name__)


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
        self.unk_id = src_dict.unk_index

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
        # sometimes num_words > lengths; this is an unresolved bug;
        # reproducible with max_tokens = 8000 on transformer base for iwslt de-en
        # temp fix is to clamp anything greater than length to length
        # TODO: investigate this further
        if torch.any(num_words > lengths):
            logger.info("SwithOut: num_words > lengths. Clamping tensor to a ceil of lengths.")
            num_words = num_words.float()
            lengths = lengths.float()
            num_words[num_words > lengths] = lengths[num_words > lengths]

        # sample the corrupted positions
        corrupt_pos = (
            num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)
        )

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

    def raml(self, sents, tau=0.1):
        bsz, n_steps = sents.size()

        # we don't want the tau to be dynamic
        if self.raml_tau is None:
            self.raml_tau = tau
        # compute mask for sents without  bos/eos/pad
        mask = torch.eq(sents, self.bos_id) | torch.eq(sents, self.eos_id) | torch.eq(sents, self.pad_id)
        lengths = (1.0 - mask.float()).sum(dim=1)

        # sample the number of words to corrupt fr each sentence
        logits = torch.arange(n_steps)
        logits = logits.float().mul_(-1).unsqueeze(0).expand_as(sents).contiguous().masked_fill_(mask, -float("inf"))
        logits = Variable(logits)  # adding to computation graph node
        probs = torch.nn.functional.softmax(logits.mul_(self.raml_tau), dim=1)
        num_words = torch.distributions.Categorical(probs).sample()
        # sometimes num_words > lengths; this is an unresolved bug;
        # reproducible with max_tokens = 8000 on transformer base for iwslt de-en
        # temp fix is to clamp anything greater than length to length
        # TODO: investigate this further
        if torch.any(num_words > lengths):
            logger.info("SwithOut: num_words > lengths. Clamping tensor to a ceil of lengths.")
            num_words = num_words.float()
            lengths = lengths.float()
            num_words[num_words > lengths] = lengths[num_words > lengths]

        # sample the corrupted positions
        corrupt_pos = (
            num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)
        )

        corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).bool()
        total_words = int(corrupt_pos.sum())

        # sample the corrupted values to add to sents
        corrupt_val = torch.LongTensor(total_words)
        # starts from 2 because pad_idx = 1, eos_idx = 2 in fairseq dict
        # we don't want to replace tokens with bos/eos/pad token
        corrupt_val = corrupt_val.random_(2, self.tgt_vocab_size)
        corrupts = torch.zeros(bsz, n_steps).long()
        corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
        sampled_sents = sents.add(Variable(corrupts)).remainder_(self.tgt_vocab_size)

        return sampled_sents

    def raml_together(self, tgt_sents, shift_tgt_sents, tau=0.1):
        def get_mask_and_lengths(sents):
            mask = torch.eq(sents, self.bos_id) | torch.eq(sents, self.eos_id) | torch.eq(sents, self.pad_id)
            lengths = (1.0 - mask.float()).sum(dim=1)
            return mask, lengths

        assert tgt_sents.size() == shift_tgt_sents.size()

        bsz, n_steps = tgt_sents.size()

        # we don't want the tau to be dynamic
        if self.raml_tau is None:
            self.raml_tau = tau
        # compute mask for sents without  bos/eos/pad
        t_mask, t_lengths = get_mask_and_lengths(tgt_sents)
        shift_t_mask, shift_t_lengths = get_mask_and_lengths(shift_tgt_sents)

        # sample the number of words to corrupt for each sentence from tgt_sentences only
        logits = torch.arange(n_steps)
        logits = (
            logits.float().mul_(-1).unsqueeze(0).expand_as(tgt_sents).contiguous().masked_fill_(t_mask, -float("inf"))
        )
        logits = Variable(logits)  # adding to computation graph node
        probs = torch.nn.functional.softmax(logits.mul_(self.raml_tau), dim=1)
        num_words = torch.distributions.Categorical(probs).sample()

        # sometimes num_words > lengths; this is an unresolved bug;
        # reproducible with max_tokens = 8000 on transformer base for iwslt de-en
        # temp fix is to clamp anything greater than length to length
        # TODO: investigate this further
        if torch.any(num_words > t_lengths):
            logger.info("SwithOut: num_words > lengths. Clamping tensor to a ceil of lengths.")
            num_words = num_words.float()
            t_lengths = t_lengths.float()
            num_words[num_words > t_lengths] = t_lengths[num_words > t_lengths]

        # sample corrupted positions
        # tgt_sents and shift_tgt_sents are of same shape
        corrupt_pos = num_words.data.float().div_(t_lengths).unsqueeze(1).expand_as(tgt_sents).contiguous()

        # sample the corrupted positions for tgt_sents
        t_corrupt_pos = corrupt_pos.masked_fill(t_mask, 0)  # don't use masked_fill_ it fills self tensor
        t_corrupt_pos = torch.bernoulli(t_corrupt_pos, out=t_corrupt_pos).bool()

        # sample the corrupted positions for shift_tgt_sents
        shift_t_corrupt_pos = corrupt_pos.masked_fill(shift_t_mask, 0)  # don't use masked_fill_ it fills self tensor
        shift_t_corrupt_pos = torch.bernoulli(shift_t_corrupt_pos, out=shift_t_corrupt_pos).bool()

        import ipdb

        ipdb.set_trace()
        # ensure there are same number of corrupt positions
        assert (
            t_corrupt_pos.sum() == shift_t_corrupt_pos.sum()
        ), "RAML: tgt and shift_tgt have different number of corrupt_pos"

        total_words = int(t_corrupt_pos.sum())

        # sample the corrupted values to add to sents
        corrupt_val = torch.LongTensor(total_words)
        # starts from 2 because pad_idx = 1, eos_idx = 2 in fairseq dict
        # we don't want to replace tokens with bos/eos/pad token
        corrupt_val = corrupt_val.random_(2, self.tgt_vocab_size)

        corrupts = torch.zeros(bsz, n_steps).long()
        t_corrupts = corrupts.masked_scatter_(t_corrupt_pos, corrupt_val)
        corrupts = torch.zeros(bsz, n_steps).long()
        shift_t_corrupts = corrupts.masked_scatter_(shift_t_corrupt_pos, corrupt_val)

        tgt_sampled_sents = tgt_sents.add(Variable(t_corrupts)).remainder_(self.tgt_vocab_size)
        shift_tgt_sampled_sents = shift_tgt_sents.add(Variable(shift_t_corrupts)).remainder_(self.tgt_vocab_size)

        return tgt_sampled_sents, shift_tgt_sampled_sents

    def word_dropout(self, sents, tau=0.1):
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
        # sometimes num_words > lengths; this is an unresolved bug;
        # reproducible with max_tokens = 8000 on transformer base for iwslt de-en
        # temp fix is to clamp anything greater than length to length
        # TODO: investigate this further
        if torch.any(num_words > lengths):
            logger.info("SwithOut: num_words > lengths. Clamping tensor to a ceil of lengths.")
            num_words = num_words.float()
            lengths = lengths.float()
            num_words[num_words > lengths] = lengths[num_words > lengths]

        # sample the corrupted positions
        corrupt_pos = (
            num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)
        )

        corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).bool()
        total_words = int(corrupt_pos.sum())

        # sample the corrupted values to add to sents
        # for word_dropout, the corrupt candidateas are always UNK.
        # recall that word_dropout is a special case of SwitchOut
        corrupt_val = torch.ones(total_words).long() * -1  # self.unk_id
        # starts from 2 because pad_idx = 1, eos_idx = 2 in fairseq dict
        # we don't want to replace tokens with bos/eos/pad token
        # corrupt_val = corrupt_val.random_(2, self.src_vocab_size)
        corrupts = torch.zeros(bsz, n_steps).long()
        corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
        sents[corrupts == -1] = self.unk_id
        # sampled_sents = torch.empty_like(sents).copy_(sents)
        sampled_sents = sents

        return sampled_sents
