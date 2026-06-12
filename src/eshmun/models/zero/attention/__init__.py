from eshmun.models.zero.attention.base import BaseAttentionModule
from eshmun.models.zero.attention.mha import MultiHeadAttention
from eshmun.models.zero.attention.gqa import GroupedQueryAttention
from eshmun.models.zero.attention.gated import GatedAttention
from eshmun.models.zero.attention.qwen import QwenGatedAttention
from eshmun.models.zero.configuration import EshmunZeroConfig

ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "sliding_window": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
    "gated": GatedAttention,
    "qwen": QwenGatedAttention
}


def build_attention(config: EshmunZeroConfig) -> BaseAttentionModule:
    attn_cls = ATTENTION_REGISTRY.get(config.attn_impl)
    if attn_cls is None:
        raise ValueError(
            f"Unknown attn_impl '{config.attn_impl}'. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    return attn_cls(config)
