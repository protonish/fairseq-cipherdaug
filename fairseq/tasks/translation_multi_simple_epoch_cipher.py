import torch
import logging
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch_eval import TranslationMultiSimpleEpochEvalTask


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch_cipher")
class TranslationMultiSimpleEpochCipherTask(TranslationMultiSimpleEpochEvalTask):
    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochEvalTask.add_args(parser)
        parser.add_argument("--reg-alpha", default=0, type=int)

    def __init__(self, args, langs, dicts, training):
        super().__init__(self, args, langs, dicts, training)
        self.criterion_reg_alpha = getattr(args, "reg_alpha", 0)

    # def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
    #     model.train()
    #     model.set_num_updates(update_num)
    #     with torch.autograd.profiler.record_function("forward"):
    #         loss, sample_size, logging_output = criterion.forward_reg(
    #             model, sample, optimizer, self.criterion_reg_alpha, ignore_grad
    #         )
    #         return loss, sample_size, logging_output
