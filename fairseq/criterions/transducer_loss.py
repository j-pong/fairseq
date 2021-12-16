# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from dataclasses import dataclass
from logging import logProcesses

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from warp_rnnt import rnnt_loss


@dataclass
class TransducerCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

class Sequence(object):
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp
@register_criterion("transducer", dataclass=TransducerCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.padding_idx = task.tgt_dict.pad() 

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        n_correct, total = self.compute_accuracy(model, net_output, sample)
        logging_output["n_correct"] = utils.item(n_correct.data)
        logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def _greedy_search(self, x, T):
        y_batch = []
        B = len(x)
        enc_out = self.encoder.forward(x)
        U_max = 300

        for b in range(B):
            t = 0
            u = 0
            y = [self.predictor.start_symbol]

            pred_state = self.predictor.initial_state.unsqueeze(0)

            while t < T[b] and u < U_max:

                pred_input = torch.tensor([y[-1]]).to(x.device)
                g_u, pred_state = self.predictor.forward_one_step(pred_input, pred_state)
                f_t = enc_out[b, t]
                h_t_u = self.joint.forward(f_t, g_u)

                argmax = h_t_u.max(-1)[1].item()

                if argmax == 0:
                    t += 1
                else:    # argmax is a label
                    u += 1
                    y.append(argmax)

            y_batch.append(y[1:])    # except start symbol

        return y_batch

    def beam_search(self, x, T, W):

        B = [Sequence(blank=0)]
        batch_size = len(x)
        enc_out = self.encoder.forward(x)
        enc_out = enc_out.squeeze() # batch_size 1 일 때 코드
        for i, f_t in enumerate(enc_out):

            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B       # Hypotheses that has emitted null from frame 't' and are now in 't+1'
            B = []      # Hypotheses that has not yet emitted null from frame 't' and so can continue emitting more symbols from 't'

            pred_state = self.predictor.initial_state.unsqueeze(0)

            while True:
                y_hat = max(A, key=lambda a: a.logp) # y^ most probable in A

                A.remove(y_hat) # remove y^ from A

                pred_input = torch.tensor([y_hat.k[-1]]).to(x.device)

                g_u, pred_state = self.predictor.forward_one_step(pred_input, y_hat.h)

                h_t_u = self.joint(f_t, g_u[0]) # g_u -> [120(out_dim)] , h_t_u -> [29(vocab)]

                logp = F.log_softmax(h_t_u, dim=0)  # pr(y^) = ~~~~

                for k in range(len(logp)):
                    yk = Sequence(y_hat)

                    yk.logp += float(logp[k]) # pr(y^+k) = ~~~~

                    if k == 0:
                        B.append(yk) # add y^ to B
                        continue

                    yk.h = pred_state; yk.k.append(k);

                    A.append(yk)    # add y^ + k to A

                y_hat = max(A, key=lambda a: a.logp)   # elements most probable in A

                yb = max(B, key=lambda a: a.logp)   # elements most probable in B

                if len(B) >= W and yb.logp >= y_hat.logp: break

            sorted(B, key=lambda a: a.logp, reverse=True)

        return [(B[0].k)[1:]]

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        target = target [:, :-1]
        frames_lengths = net_output[0]["input_lengths"]
        labels_lengths = sample["target_lengths"] - 1 # target label length is U - 1

        loss = rnnt_loss(
            lprobs.float(), 
            target.int(), 
            frames_lengths=frames_lengths.int(), 
            labels_lengths=labels_lengths.int(),
            reduction="sum",
            blank=self.padding_idx
        )

        return loss, loss
    
    def compute_accuracy(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        mask = target.ne(self.padding_idx) # B, U

        bsz, tsz, usz, dsz = lprobs.size()
        max_score = lprobs.max(-1).values
        max_score_idx = max_score.argmax(-2) # B, U
        max_score_idx = max_score_idx.view(bsz * usz)

        label_matrix = lprobs.argmax(-1).permute(0, 2, 1) # B, U, T
        label_matrix = label_matrix.view(bsz * usz, tsz)
        target_pred = label_matrix[torch.arange(bsz * usz), max_score_idx].view(bsz, usz)

        n_correct = torch.sum(
            target_pred.masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
