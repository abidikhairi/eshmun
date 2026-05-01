"""
Configuration for Eshmun-Zero.

Eshmun-Zero is the first model in the Eshmun series, featuring a parallel
local + global self-attention layer combined via a single learnable gate alpha,
where beta = 1 - alpha.
"""

from transformers import PretrainedConfig


class EshmunZeroConfig(PretrainedConfig):
    r"""
    Configuration class for Eshmun-Zero.

    Args:
        vocab_size (`int`, *optional*, defaults to 25):
            Vocabulary size of the model.
        embedding_size (`int`, *optional*, defaults to 4096)
            Dimensionality of the embedding and lm_head layers.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward layer.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function in the feed-forward layer.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            Dropout probability for all fully connected layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            Dropout probability for attention weights.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            Maximum sequence length the model can handle.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Std of the truncated normal initializer for weight initialization.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Epsilon for layer normalization.
        local_window_size (`int`, *optional*, defaults to 64):
            Fixed window size for local self-attention. Each token attends to
            `local_window_size` tokens symmetrically around it.
        apply_random_mask (`bool`, *optional*, defaults to `False`):
            Whether to apply random mask along with local_attention_mask.
        alpha_init (`float`, *optional*, defaults to 0.5):
            Initial value for the learnable gate alpha (before sigmoid).
            A value of 0.0 means alpha = sigmoid(0) = 0.5 at init, giving
            equal weight to local and global attention.
        pad_token_id (`int`, *optional*, defaults to 0):
            Id of the padding token.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Only `"absolute"` is supported.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder (causal LM).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/value attentions (KV cache).
            Only relevant when `is_decoder=True`.

    Example::

        >>> from eshmun.models.zero import EshmunZeroConfig, EshmunZeroModel
        >>> config = EshmunZeroConfig()
        >>> model = EshmunZeroModel(config)
    """

    model_type = "eshmun_zero"

    def __init__(
        self,
        vocab_size: int = 25,
        embedding_size: int = 4096,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "silu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 514,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        local_window_size: int = 12,
        apply_random_mask: bool = False,
        alpha_init: float = 0.0,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        is_decoder: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.local_window_size = local_window_size
        self.apply_random_mask = apply_random_mask
        self.alpha_init = alpha_init
        self.position_embedding_type = position_embedding_type
        self.is_decoder = is_decoder
        self.use_cache = use_cache
