from typing import Literal
from transformers import PreTrainedConfig


class EshmunZeroConfig(PreTrainedConfig):
    model_type = 'eshmun_zero'
    
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 256,
        intermediate_size: int = 768,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        dropout: float = 0.05,
        max_seq_len: int = 512,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
        attn_impl: Literal[
            "mha", "sparse", "sliding_window", "gqa", "compressed", "gated",
            "qwen", "token_filter"
        ] = "mha",
        use_rope: bool = False,
        top_k: int = 5,
        num_kv_heads: int = 4,
        window_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.attn_impl = attn_impl
        self.use_rope = use_rope
        self.top_k = top_k
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size

        ## NOTE: for compatibility with .generate() api
        self.num_hidden_layers = num_layers
