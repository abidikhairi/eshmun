import torch
from torch import nn

from eshmun.models.zero.configuration import EshmunZeroConfig


class MLP(nn.Module):
    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        
        self.dropout = config.dropout
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, False)
        self.silu = nn.SiLU()
        self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size, False)
        
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(hidden_states) * self.silu(self.gate_proj(hidden_states))
        hidden_states = self.out_proj(hidden_states)
        
        if self.training:
            hidden_states = torch.dropout(hidden_states, p=self.dropout, train=self.training)
        
        return hidden_states
