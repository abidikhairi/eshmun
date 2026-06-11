import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from eshmun.models.zero.configuration import EshmunZeroConfig
from eshmun.models.zero.attention import build_attention
from eshmun.models.zero.mlp import MLP


class EshmunZeroLayer(nn.Module):
    def __init__(self, config: EshmunZeroConfig, layer_idx: int):
        super().__init__()

        self.pre_attn_norm = nn.RMSNorm(config.hidden_size)
        self.self_attn = build_attention(config)
        self.pre_mlp_norm = nn.RMSNorm(config.hidden_size)

        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class EshmunZeroDecoder(nn.Module):
    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        self.attn_impl = config.attn_impl

        self.layers = nn.ModuleList(
            [
                EshmunZeroLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        for layer in self.layers:
            if self.attn_impl in ("sliding_window"):
                # NOTE: in sliding window attn, we use sliding_mask as the causal attention mask
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=sliding_window_mask,
                    sliding_window_mask=sliding_window_mask,
                    **kwargs,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    **kwargs,
                )

        return hidden_states


class EshmunZero(PreTrainedModel):

    config_class = EshmunZeroConfig

    _tied_weights_keys = {"lm_head.weight": "tokens_embed.weight"}

    def __init__(self, config: EshmunZeroConfig):
        super().__init__(config)

        self.tokens_embed = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.positions_embed = nn.Embedding(config.max_seq_len, config.hidden_size)

        self.model = EshmunZeroDecoder(config)

        self.final_layer_norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, False)

        ## Tie weights explicitly
        self.lm_head.weight = self.tokens_embed.weight

        self.generation_config = GenerationConfig(
            do_sample=True,
            top_k=950,
            top_p=0.95,
            repetition_penalty=1.3,
        )

        self.post_init()

    def _build_3d_mask(self, attention_mask: torch.Tensor):
        bsz, seq_len = attention_mask.shape
        expanded_attention_mask = attention_mask[:, None, None, :]
        full_mask = torch.ones(seq_len, seq_len).to(expanded_attention_mask.device)

        expanded_attention_mask = expanded_attention_mask * full_mask
        expanded_attention_mask = (1.0 - expanded_attention_mask) * torch.finfo(
            torch.float32
        ).min

        return expanded_attention_mask

    def _build_sliding_window_mask(self, attention_mask: torch.Tensor):
        bsz, seq_len = attention_mask.shape
        expanded_attention_mask = attention_mask[:, None, None, :]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(
            expanded_attention_mask.device
        )
        too_far_mask = torch.tril(
            torch.ones(seq_len, seq_len), diagonal=-(self.config.window_size + 1)
        ).to(expanded_attention_mask.device)

        sliding_window_mask = causal_mask - too_far_mask
        sliding_window_mask_expanded = expanded_attention_mask * sliding_window_mask
        sliding_window_mask_expanded = (
            1.0 - sliding_window_mask_expanded
        ) * torch.finfo(self.dtype).min

        return sliding_window_mask_expanded

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len).unsqueeze_(0).repeat(bsz, 1).to(input_ids.device)
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        sliding_window_mask = None
        if self.config.attn_impl in ("gated", "sliding_window"):
            sliding_window_mask = self._build_sliding_window_mask(attention_mask)

        attention_mask_expanded = self._build_3d_mask(attention_mask)

        tokens_embeds = self.tokens_embed(input_ids)
        positions_embed = self.positions_embed(position_ids)

        hidden_states = tokens_embeds + positions_embed

        last_hidden_states = self.model(
            hidden_states, attention_mask_expanded, sliding_window_mask
        )

        hidden_states = self.final_layer_norm(last_hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)
