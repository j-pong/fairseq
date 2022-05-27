import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq.models import BaseFairseqModel, register_model

# Mode load related
import re

from fairseq import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

## Wav2vec2 model related
from fairseq.models.wav2vec.wav2vec2 import (
    Wav2Vec2Config, 
    Wav2Vec2Model
)

@dataclass
class CtrlWav2Vec2Config(Wav2Vec2Config):
    no_ctrl_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    ctrl_w2v_path: str = field(
        default="", metadata={"help": "path to wav2vec 2.0 model"}
    )
    ctrl_type: str = field(
        default="lwf", metadata={"help": "type of CTRL's transfer loss"}
    )


@register_model("ctrlwav2vec2", dataclass=CtrlWav2Vec2Config)
class CtrlWav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg: CtrlWav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        # Manually fix the configuration
        arg_overrides = {}

        if cfg.ctrl_w2v_path is not None and not cfg.no_ctrl_pretrained_weights:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.ctrl_w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            # w2v_args.criterion = None
            # w2v_args.lr_scheduler = None
            # cfg.w2v_args = w2v_args

            # logging.info(w2v_args)
        else:
            state = None
        
        model = Wav2Vec2Model(cfg)
        model_anchor = Wav2Vec2Model(cfg)
        self.ctrl_type = cfg.ctrl_type

        if state is not None and not cfg.no_ctrl_pretrained_weights:
            self.load_model_weights(state, model, cfg)
            self.load_model_weights(state, model_anchor, cfg)
            logging.info(f"Pre-trained representation model is loaded from {cfg.ctrl_w2v_path}")

        self.model = model
        self.model_anchor = model_anchor
        self.model_anchor.requires_grad_(False)
        self.model_anchor.eval()
        assert not self.model_anchor.training 

        self.num_updates = 0

    def load_model_weights(self, state, model, cfg):
        # filtering for data2vec
        # if "_ema" in state["model"]:
        #     del state["model"]["_ema"]

        # load from CtrlModel
        r = re.compile("model_anchor\.")
        filtered_list = list(filter(r.match, state["model"].keys()))

        new_big_dict = {
                k.replace("model.", "") if "model." in k else k: v 
                for (k, v) in state["model"].items() if k not in filtered_list
            }

        # load model from state
        model.load_state_dict(new_big_dict, strict=True)
    
    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        return cls(cfg)

    def get_logits(self, net_output):
        return self.model.get_logits(net_output)

    def get_targets(self, sample, net_output):
        return self.model.get_targets(sample, net_output)

    def get_extra_losses(self, net_output):
        loss_extra = self.model.get_extra_losses(net_output)
        if self.ctrl_type == "l2":
            anchor_params = {name : p for name, p in self.model_anchor.named_parameters()}

            loss = 0.0
            n_grad_param = 0
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    loss += F.mse_loss(p.float(), anchor_params[name].float())
                    n_grad_param += 1
            loss = (loss / n_grad_param).type_as(p)
        elif self.ctrl_type == "lwf":
            raise NotImplementedError
            inputs = self.modle.get_logits(net_output)
            targets = net_output["logits_target"].detach()

            loss = F.kl_div(inputs.float().log_softmax(1), targets.float().softmax(1), reduction="none").sum(1)

            mask = torch.logical_or(torch.isinf(loss), torch.isnan(loss))
            loss = loss.masked_fill(mask, 0.0)
            loss = (loss / (~mask).float().sum()).sum().type_as(inputs)
        elif self.ctrl_type == "ewc":
            raise NotImplementedError
        else:
            raise AttributeError

        loss_extra.append(loss)

        return loss_extra

    def forward(self, **kwargs):
        net_output = self.model(**kwargs)
        
        if self.ctrl_type == "lwf":
            raise NotImplementedError
            self.model_anchor.training = False
            with torch.no_grad():
                net_output_anchor = self.model_anchor(
                    **kwargs,
                    mask=False,
                    mask_indices=net_output["mask_indices"],
                    negs=net_output["negs"],
                    cb_negs=net_output["cb_negs"]
                )
            net_output["logits_target"] = self.model_anchor.get_logits(net_output_anchor) 
        else:
            pass

        return net_output

    def remove_pretraining_modules(self):
        self.anchor_model = None
        self.model.remove_pretraining_modules()
        
        logging.info("Remove the pretraining modules and model init parameter from ASR model w2v_path")
    

