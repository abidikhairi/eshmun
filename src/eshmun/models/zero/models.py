"""
Eshmun-Zero model definitions (HuggingFace-compatible).

Models:
  - EshmunModel          : base encoder (no LM head)
  - EshmunForMaskedLM    : encoder + MLM head (BERT-style, bidirectional)
  - EshmunForCausalLM    : decoder + causal LM head (GPT-style, autoregressive)

All models inherit from PreTrainedModel and expose the standard HF interface:
  - from_pretrained / save_pretrained
  - gradient checkpointing hooks
  - standard forward signature with attention_mask, labels, etc.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
)

from eshmun.models.zero.configuration import EshmunZeroConfig
from eshmun.models.zero.components import (
    EshmunEmbeddings,
    EshmunEncoder,
    EshmunPooler,
)

# ---------------------------------------------------------------------------
# Utility: build additive attention mask from HF-style padding mask
# ---------------------------------------------------------------------------


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Convert a HF-style attention mask (1 = attend, 0 = ignore) to an
    additive mask (0 = attend, -inf = ignore) broadcastable over heads.

    Input:  (B, T) with values in {0, 1}
    Output: (B, 1, 1, T) with values in {0.0, -inf}
    """
    if attention_mask is None:
        return None

    mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
    mask = (1.0 - mask.to(dtype)) * torch.finfo(dtype).min
    return mask


# ---------------------------------------------------------------------------
# Base Model (encoder only, no LM head)
# ---------------------------------------------------------------------------


class EshmunZeroPreTrainedModel(PreTrainedModel):
    """Abstract base with weight initialization."""

    config_class = EshmunZeroConfig
    base_model_prefix = "eshmun_zero"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class EshmunZeroModel(EshmunZeroPreTrainedModel):
    """
    Eshmun-Zero base model: embeddings + bidirectional encoder (no LM head).

    Suitable as a backbone for downstream tasks (classification, NER, etc.)
    or as the encoder in EshmunForMaskedLM.

    Args:
        config: EshmunConfig
        add_pooling_layer: whether to include a [CLS] pooler (default True)

    Outputs (BaseModelOutputWithPooling):
        last_hidden_state: (B, T, D)
        pooler_output: (B, D) if add_pooling_layer else None
        alphas: list of per-layer gate values if output_alphas=True
    """

    def __init__(self, config: EshmunZeroConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.embeddings = EshmunEmbeddings(config)
        self.encoder = EshmunEncoder(config)
        self.pooler = EshmunPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.wte

    def set_input_embeddings(self, value: nn.Embedding):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_alphas: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutputWithPooling, Tuple]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide either input_ids or inputs_embeds, not both.")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

        additive_mask = _prepare_attention_mask(attention_mask, inputs_embeds.dtype)

        hidden_states, alphas = self.encoder(
            inputs_embeds,
            attention_mask=additive_mask,
            output_alphas=output_alphas,
        )

        pooled = self.pooler(hidden_states) if self.pooler is not None else None

        if not return_dict:
            return (hidden_states, pooled) + ((alphas,) if output_alphas else ())

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=None,
            attentions=None,
        )


# ---------------------------------------------------------------------------
# MLM head components
# ---------------------------------------------------------------------------


class EshmunMLMPredictionHead(nn.Module):
    """
    Standard BERT-style MLM prediction head:
        Dense -> Act -> LayerNorm -> Decoder (tied to embedding weights)
    """

    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # decoder is tied to input embeddings — bias is separate
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


# ---------------------------------------------------------------------------
# Masked Language Model
# ---------------------------------------------------------------------------


class EshmunForMaskedLM(EshmunZeroPreTrainedModel):
    """
    Eshmun-Zero for Masked Language Modelling (BERT-style).

    The model is bidirectional: each token attends to all others (global)
    or a symmetric local window (local), combined via alpha gate.

    Usage::

        >>> config = EshmunZeroConfig()
        >>> model = EshmunZeroForMaskedLM(config)
        >>> input_ids = torch.randint(0, config.vocab_size, (2, 128))
        >>> labels = input_ids.clone()
        >>> out = model(input_ids=input_ids, labels=labels)
        >>> out.loss.backward()

    Args:
        config: EshmunConfig

    Inputs:
        input_ids: (B, T)
        attention_mask: (B, T) with 1=attend, 0=ignore (optional)
        token_type_ids: (B, T) (optional)
        position_ids: (B, T) (optional)
        inputs_embeds: (B, T, D) alternative to input_ids (optional)
        labels: (B, T) token ids; positions with -100 are ignored in the loss
        output_alphas: bool — return per-layer alpha gate values
        return_dict: bool

    Outputs (MaskedLMOutput):
        loss: scalar cross-entropy (only if labels provided)
        logits: (B, T, vocab_size)
        alphas: list[Tensor] if output_alphas
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: EshmunZeroConfig):
        super().__init__(config)
        self.eshmun = EshmunZeroModel(config, add_pooling_layer=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.eshmun.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.eshmun.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_alphas: bool = False,
        return_dict: bool = True,
    ) -> Union[MaskedLMOutput, Tuple]:

        outputs = self.eshmun(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_alphas=output_alphas,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # (B, T, D)
        logits = self.lm_head(sequence_output)  # (B, T, V)

        loss = torch.FloatTensor(-1.0)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,)
            if output_alphas:
                out += (outputs.get("alphas"),)
            return ((loss,) + out) if loss is not None else out

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
