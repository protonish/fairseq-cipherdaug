import numpy as np
import torch
from fairseq.data import data_utils


class SwitchOut(object):
    def __init__(self, src_dict, tgt_dict) -> None:
        super().__init__()
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.src_vocab_size = self.src_dict.__len__()
        self.tgt_vocab_size = self.tgt_dict.__len__()

        bos_id = self.dictionary.bos_index
        eos_id = self.dictionary.eos_index
        pad_id = self.dictionary.pad_index

        self.switch_tau = None
        self.raml_tau = None

    def switchout(self, batch):
        sents = batch["net_input"]["src_tokens"]

        return None
