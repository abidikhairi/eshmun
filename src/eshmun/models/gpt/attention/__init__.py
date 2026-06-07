from eshmun.models.gpt.attention.base import BaseAttentionModule
from eshmun.models.gpt.attention.mha import MultiHeadAttention
from eshmun.models.gpt.attention.gqa import GroupedQueryAttention
from eshmun.models.gpt.attention.gated import GatedAttention
from eshmun.models.gpt.gpt_configuration import EshmunGPTConfig

ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "sliding_window": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
    "gated": GatedAttention,
}


def build_attention(config: EshmunGPTConfig) -> BaseAttentionModule:
    attn_cls = ATTENTION_REGISTRY.get(config.attn_impl)
    if attn_cls is None:
        raise ValueError(
            f"Unknown attn_impl '{config.attn_impl}'. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    return attn_cls(config)
