import math
from abc import ABC, abstractmethod
import torch
from torch import nn
from eshmun.models.gpt.gpt_configuration import EshmunGPTConfig



class BaseAttentionModule(nn.Module, ABC):
    def __init__(self, config: EshmunGPTConfig):
        super().__init__()
        self._validate_config(config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        
        self.scaling_factor = 1.0 / math.sqrt(self.head_dim)

    def _validate_config(self, config):
        assert config.hidden_size % config.num_attention_heads == 0, """
            hidden_size should be divisible by num_attention_heads, got {} and {}
        """.format(config.hidden_size, config.num_attention_heads)
    
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        pass
    
    