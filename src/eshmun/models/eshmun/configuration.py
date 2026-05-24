"""Eshmun model configuration"""
from transformers.configuration_utils import PreTrainedConfig


class EshmunConfig(PreTrainedConfig):
    r"""
    Configuration class for the Eshmun model.

    Args:
        vocab_size (`int`, *optional*, defaults to 50272):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length the model can handle.
        do_layer_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to perform layer normalization before the attention block.
        word_embed_proj_dim (`int`, *optional*, defaults to `hidden_size`):
            Dimensionality of the word embeddings. Can be set smaller than `hidden_size`
            to down-project embeddings.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for all fully connected layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads per decoder layer.
        activation_function (`str`, *optional*, defaults to `"relu"`):
            Activation function in the feed-forward layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            LayerDrop probability. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556).
        init_std (`float`, *optional*, defaults to 0.02):
            Std of the truncated normal initializer for weight initialization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/value attentions (KV cache).
        enable_bias (`bool`, *optional*, defaults to `True`):
            Whether or not if the linear layers in the attention blocks should use the bias term.
        layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not if the layer norms should have learnable parameters.

    Example:

    ```python
    >>> from eshmun.models.eshmun import EshmunConfig, EshmunModel

    >>> # Initializing an Eshmun configuration
    >>> configuration = EshmunConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = EshmunModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "eshmun"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 50272
    hidden_size: int = 768
    num_hidden_layers: int = 12
    ffn_dim: int = 3072
    max_position_embeddings: int = 2048
    do_layer_norm_before: bool = True
    _remove_final_layer_norm: bool = False
    word_embed_proj_dim: int = 768
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    num_attention_heads: int = 12
    activation_function: str = "relu"
    layerdrop: float | int = 0.0
    init_std: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 2
    enable_bias: bool = True
    layer_norm_elementwise_affine: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.word_embed_proj_dim = (
            self.word_embed_proj_dim
            if self.word_embed_proj_dim is not None
            else self.hidden_size
        )
        super().__post_init__(**kwargs)


__all__ = ["EshmunConfig"]
