# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import torch

from . import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


class AddTargetDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        state,
        pad,
        eos,
        blank,
        split,
        batch_targets,
        pad_state=-1,
        process_label=None,
        label_len_fn=None,
        add_to_input=False,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.state = state
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.blank = blank # 0
        self.split = split # 4
        self.pad_state = pad_state
        self.process_label = process_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input
        self.text_compressor = TextCompressor(level=text_compression_level)

    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)
    
    def get_label_and_duration(self, index):
        state = self.state[index]
        state = torch.tensor([int(s) for s in state.replace("\n", "").split(" ")])
        state = state.long()

        toks, count = state.unique_consecutive(return_counts=True)

        # 1. split to word level segments
        toks, count = self.split_tensor(toks, self.split, count)
        count = torch.tensor(count).cumsum(dim=-1)

        # 2. random grouping
        n = len(toks)
        if n <= 5:
            n_g = 5
            toks = [torch.cat(toks)]
            count = torch.tensor([count[-1]])
        else:
            n_g = random.randint(5, n)
            if n_g != n:
                toks = self.group_target(toks, n_g)
                count = self.group_duration(count, n_g)
        assert len(toks) == len(count)
        count = torch.cat([torch.tensor([0]), count])

        return toks, count

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index, process_fn=self.process_label)
        if self.state is not None:
            item["label"], item["duration"] = self.get_label_and_duration(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = self.label_len_fn(self.get_label(index))
        return sz, own_sz

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        if self.state is not None:
            new_target = []
            for ta in target:
                new_target = new_target + ta
            target = new_target
            duration = [s["duration"] for s in samples if s["id"] in indices]

        if self.add_to_input:
            eos = torch.LongTensor([self.eos])
            prev_output_tokens = [torch.cat([eos, t], axis=-1) for t in target]
            target = [torch.cat([t, eos], axis=-1) for t in target]
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
            if getattr(collated["net_input"], "prev_output_tokens", None):
                collated["net_input"]["prev_output_tokens"] = data_utils.collate_tokens(
                    collated["net_input"]["prev_output_tokens"],
                    pad_idx=self.pad,
                    left_pad=False,
                )
            if self.state is not None:
                duration = data_utils.collate_tokens(duration, pad_idx=-1, left_pad=False)
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target
        if self.state is not None:
            collated["net_input"]["duration"] = duration
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
    
    def split_tensor(self, tensor, split_value, subtensor=None):
        split_indices = (tensor == split_value).nonzero(as_tuple=True)[0]
        split_indices = torch.cat([
            torch.tensor([-1]), 
            split_indices, 
            torch.tensor([len(tensor)])
        ])
        split_lengths = split_indices[1:] - split_indices[:-1] - 1
        
        split_tensors = []
        split_subtensors = []
        start_idx = 0
        for length in split_lengths:
            if length > 0:
                ts = tensor[start_idx : start_idx + length + 1]
                ts = ts[(ts != self.blank) & (ts != self.pad)]
                if len(ts) == 0:
                    ts = torch.tensor([self.blank])
                split_tensors.append(ts) # for pseudo-label
                
                if subtensor is not None:
                    split_subtensors.append(subtensor[start_idx : start_idx + length + 1].sum()) # NOTE: for duration

            start_idx += length + 1

        if subtensor is not None:
            return split_tensors, split_subtensors
        else:
            return split_tensors

    def group_target(self, input_list, n):
        return [torch.cat(input_list[i:i + n]) for i in range(0, len(input_list), n)]
    
    def group_duration(self, tensor, n):
        return torch.tensor([tensor[i:i + n][-1] for i in range(0, len(tensor), n)])