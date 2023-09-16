import sys

from dataclasses import  is_dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from fairseq import distributed_utils, utils
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

from examples.speech_recognition.new.infer import config_path, InferConfig, main, InferenceProcessor
from examples.speech_recognition.new.infer import reset_logging, logger, progress_bar

import numpy as np

from aligner import BypassSegmentor as Segmentor

class InferenceProcessorSeg(InferenceProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Task inferene_step warpper
        self.dur_type = "accum" # or "overlap"

        self.segmentor = Segmentor(self.models[0], self.tgt_dict)
        self.total_sample_size = len(self.task.dataset(self.cfg.dataset.gen_subset))
        self.manifest = {i: {} for i in range(self.total_sample_size)}

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.gen_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=(sys.maxsize, sys.maxsize),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=1, #self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        ).next_epoch_itr(shuffle=False)
    
    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.gen_timer.start()
        sample_ids = sample["id"]

        results = self.segmentor.segment(sample, prop2raw=False)

        for i, sample_id in enumerate(sample_ids):
            sample_id = sample_id.item()
            if results["ltr"] is not None :
                assert len(sample_ids) == 1
                ltr = results["ltr"].strip()
                self.manifest[sample_id]["ltr"] = ltr

            if results["dur"] is not None :
                dur = results['dur']
                if self.dur_type == "overlap":
                    dur = " ".join([f"{d[0]} {d[1]} |" for d in dur])
                else:
                    dur = " ".join([f"{d} |" for d in dur])
                self.manifest[sample_id]["dur"] = dur

            if results["state"] is not None :
                self.manifest[sample_id]["state"] = results["state"][i]

def main(cfg: InferConfig) -> float:
    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 4000000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")

    logger.info(cfg.common_eval.path)

    with InferenceProcessorSeg(cfg) as processor:
        for sample in processor:
            processor.process_sample(sample)

        processor.log_generation_time()

        if cfg.decoding.results_path is not None:
            processor.merge_shards()

        if distributed_utils.is_master(cfg.distributed_training):
            if "ltr" in processor.manifest[0]:
                ltr_out = open(f"train.ltr", 'w') 
                wrd_out = open(f"train.wrd", 'w')
            else:
                ltr_out = None
                wrd_out = None

            if "dur" in processor.manifest[0]:
                dur_out = open(f"train.dur", 'w')
            else:
                dur_out = None

            if "state" in processor.manifest[0]:
                state_out = open(f"train.state", "w")
            else:
                state_out = None

            for sample_id in sorted(processor.manifest.keys()):
                if ltr_out is not None:
                    print(processor.manifest[sample_id]["ltr"], file=ltr_out)
                    print(
                        processor.manifest[sample_id]["ltr"].replace(" ", "").replace("|", " ").strip(),
                        file=wrd_out,
                    )

                if dur_out is not None:
                    print(processor.manifest[sample_id]["dur"], file=dur_out)

                if state_out is not None:
                    s = ' '.join(map(str, processor.manifest[sample_id]["state"]))
                    print(s, file=state_out) # numpy.frombuffer(s[0].decode("utf-8").replace("\n", "").encode(), dtype=np.uint8)

        return 0.0

@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    utils.import_user_module(cfg.common)

    # logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    return wer


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
