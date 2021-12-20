#!/usr/bin/env python3

from dataclasses import dataclass
from json import encoder
import logging
import math
from re import S
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase
from fairseq.models.transformer.transformer_config import TransformerConfig, EncDecBaseConfig, QuantNoiseConfig
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
)
from torch import Tensor


logger = logging.getLogger(__name__)

from dataclasses import fields
from fairseq.utils import safe_hasattr, safe_getattr

class MultiTransformerEncoderConfig(TransformerConfig):
    @classmethod
    def from_namespace(cls, args, encoder_prefix="speech"):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            for fld in fields(cls):
                if fld.name == "encoder":
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, f"{encoder_prefix}_encoder", seen
                        )
                elif fld.name == "quant_noise":
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args


class TransformerEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, args, encoder_prefix):
        self.encoder_prefix = encoder_prefix
        super().__init__(MultiTransformerEncoderConfig.from_namespace(args, encoder_prefix=encoder_prefix))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, MultiTransformerEncoderConfig.from_namespace(args, encoder_prefix=self.encoder_prefix)
        )


class Conv1dSubsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("contra_transformer_transducer")
class ContraTransformerTransducer(BaseFairseqModel):
    def __init__(self, args, speech_encoder, speech_encoder_proj, text_encoder, text_encoder_proj, joint):
        super().__init__()

        self.speech_encoder = speech_encoder
        self.speech_encoder_proj = speech_encoder_proj
        self.text_encoder = text_encoder
        self.text_encoder_proj = text_encoder_proj
        self.joint = joint

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # speech encoder
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        # speech encoder transformer
        parser.add_argument(
            "--speech-encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--speech-encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--speech-encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--speech-encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--speech-encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        # text encoder
        parser.add_argument(
            "--text-encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--text-encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--text-encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--text-encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--text-encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )

    @classmethod
    def build_speech_encoder(cls, args):
        encoder = SpeechTransformerEncoder(args)
        # pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        # if pretraining_path is not None:
        #     if not Path(pretraining_path).exists():
        #         logger.warning(
        #             f"skipped pretraining because {pretraining_path} does not exist"
        #         )
        #     else:
        #         encoder = checkpoint_utils.load_pretrained_component_from_model(
        #             component=encoder, checkpoint=pretraining_path
        #         )
        #         logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_text_encoder(cls, args, task, embed_tokens):
        encoder = TextTransformerEncoder(args, task.target_dictionary, embed_tokens)
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        text_encoder_embed_tokens = build_embedding(
            task.target_dictionary, args.text_encoder_embed_dim
        )
        speech_encoder = cls.build_speech_encoder(args)
        text_encoder = cls.build_text_encoder(args, task, text_encoder_embed_tokens)
        speech_encoder_proj = nn.Linear(args.speech_encoder_embed_dim, args.speech_encoder_embed_dim)
        text_encoder_proj = nn.Linear(args.text_encoder_embed_dim, args.text_encoder_embed_dim)

        joint = nn.Linear(
            args.speech_encoder_embed_dim,
            len(task.target_dictionary)
        )

        return cls(args, speech_encoder, speech_encoder_proj, text_encoder, text_encoder_proj, joint)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        speech_encoder_output, text_encoder_output = net_output
        e_speech = self.speech_encoder_proj(speech_encoder_output["encoder_out"][0])
        e_text = self.text_encoder_proj(text_encoder_output["encoder_out"][0])

        logits = self.joint(
            e_speech.transpose(1, 0).unsqueeze(2) + 
            e_text.transpose(1, 0).unsqueeze(1)
        )
        logits = logits.float()

        import torch.nn.functional as F
        if log_probs:
            lprobs = F.log_softmax(logits, dim=-1)
        else:
            lprobs = F.softmax(logits, dim=-1)

        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_tokens_length):
        speech_encoder_out = self.speech_encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        text_encoder_out = self.text_encoder(src_tokens=prev_output_tokens, src_lengths=prev_output_tokens_length)
        return speech_encoder_out, text_encoder_out 

class TextTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embeded_tokens):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.text_encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()

        self.embed_tokens = embeded_tokens

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                args.text_encoder_embed_dim,
                self.padding_idx,
                learned=args.text_encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args, "text") for _ in range(args.text_encoder_layers)]
        )
        if args.text_encoder_normalize_before:
            self.layer_norm = LayerNorm(args.text_encoder_embed_dim)
        else:
            self.layer_norm = None

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x = self.embed_scale * self.embed_tokens(src_tokens)

        encoder_padding_mask = lengths_to_padding_mask(src_lengths)

        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(src_tokens)
            x += positions
        x = self.dropout_module(x)
        x = x.transpose(0, 1)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class SpeechTransformerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.speech_encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.speech_encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.speech_encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args, "speech") for _ in range(args.speech_encoder_layers)]
        )
        if args.speech_encoder_normalize_before:
            self.layer_norm = LayerNorm(args.speech_encoder_embed_dim)
        else:
            self.layer_norm = None

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "input_lengths": input_lengths,
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates



@register_model_architecture(model_name="contra_transformer_transducer", arch_name="contra_transformer_transducer")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # speech encoder 
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # speech encoder
    args.speech_encoder_embed_dim = getattr(args, "speech_encoder_embed_dim", 512)
    args.speech_encoder_ffn_embed_dim = getattr(args, "speech_encoder_ffn_embed_dim", 2048)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 6)
    args.speech_encoder_attention_heads = getattr(args, "speech_encoder_attention_heads", 8)
    args.speech_encoder_normalize_before = getattr(args, "speech_encoder_normalize_before", True)
    # text encoder
    args.text_encoder_embed_dim = getattr(args, "text_encoder_embed_dim", 512)
    args.text_encoder_ffn_embed_dim = getattr(args, "text_encoder_ffn_embed_dim", 2048)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 2)
    args.text_encoder_attention_heads = getattr(args, "text_encoder_attention_heads", 8)
    args.text_encoder_normalize_before = getattr(args, "text_encoder_normalize_before", True)
    args.text_encoder_learned_pos = getattr(args, "text_encoder_learned_pos", False)
    # transformer
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("contra_transformer_transducer", "contra_transformer_transducer_s")
def s2t_transformer_s(args):
    args.speech_encoder_embed_dim = getattr(args, "speech_encoder_embed_dim", 256)
    args.speech_encoder_ffn_embed_dim = getattr(args, "speech_encoder_ffn_embed_dim", 256 * 8)
    args.speech_encoder_attention_heads = getattr(args, "speech_encoder_attention_heads", 4)
    args.text_encoder_embed_dim = getattr(args, "text_encoder_embed_dim", 256)
    args.text_encoder_ffn_embed_dim = getattr(args, "text_encoder_ffn_embed_dim", 256 * 8)
    args.text_encoder_attention_heads = getattr(args, "text_encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("contra_transformer_transducer", "contra_transformer_transducer_m")
def s2t_transformer_m(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.speech_encoder_embed_dim = getattr(args, "speech_encoder_embed_dim", 512)
    args.speech_encoder_ffn_embed_dim = getattr(args, "speech_encoder_ffn_embed_dim", 512 * 4)
    args.speech_encoder_attention_heads = getattr(args, "speech_encoder_attention_heads", 8)
    args.text_encoder_attention_heads = getattr(args, "text_encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("contra_transformer_transducer", "contra_transformer_transducer_l")
def s2t_transformer_l(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 18)
    args.speech_encoder_embed_dim = getattr(args, "speech_encoder_embed_dim", 512)
    args.speech_encoder_ffn_embed_dim = getattr(args, "speech_encoder_ffn_embed_dim", 512 * 4)
    args.speech_encoder_attention_heads = getattr(args, "speech_encoder_attention_heads", 8)
    args.text_encoder_attention_heads = getattr(args, "text_encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("contra_transformer_transducer", "contra_transformer_transducer_xl")
def s2t_transformer_xl(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 24)
    args.speech_encoder_embed_dim = getattr(args, "speech_encoder_embed_dim", 512)
    args.speech_encoder_ffn_embed_dim = getattr(args, "speech_encoder_ffn_embed_dim", 512 * 4)
    args.speech_encoder_attention_heads = getattr(args, "speech_encoder_attention_heads", 8)
    args.text_encoder_attention_heads = getattr(args, "text_encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)

