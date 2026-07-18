"""Shared helpers for finetuning scripts."""

import torch

from eshmun.models.zero import EshmunZero


LENGTH_BUCKETS = ["0-100", "100-200", "200-300", "300-400", ">400"]


def load_model(model_path: str, device: torch.device) -> EshmunZero:
    model = EshmunZero.from_pretrained(model_path, torch_dtype=torch.float32)
    model.to(device)  # type: ignore[arg-type]
    return model


def get_hidden_states(
    model: EshmunZero, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Replicate EshmunZero.forward up to (and including) the final layer norm."""
    bsz, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(bsz, 1)

    sliding_window_mask = None
    if model.config.attn_impl in ("gated", "sliding_window"):
        sliding_window_mask = model._build_sliding_window_mask(attention_mask)

    attention_mask_expanded = model._build_3d_mask(attention_mask)

    hidden_states = model.tokens_embed(input_ids) + model.positions_embed(position_ids)
    hidden_states = model.model(hidden_states, attention_mask_expanded, sliding_window_mask)

    return model.final_layer_norm(hidden_states)


def length_bucket(length: int) -> str:
    if length < 100:
        return "0-100"
    if length < 200:
        return "100-200"
    if length < 300:
        return "200-300"
    if length < 400:
        return "300-400"
    return ">400"
