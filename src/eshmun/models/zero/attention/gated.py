import torch
from torch import nn

from eshmun.models.zero.configuration import EshmunZeroConfig
from eshmun.models.zero.attention.base import BaseAttentionModule
from eshmun.models.zero.attention.rope import RoPE


class GatedAttention(BaseAttentionModule):
    def __init__(self, config: EshmunZeroConfig):
        super().__init__(config)
        
        if self.use_rope:
            self.rope = RoPE(self.head_dim, self.config.max_seq_len, base=10_000)
        
        self.w_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.w_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.w_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_g = nn.Parameter(torch.zeros(self.num_heads))
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        
        hidden_states = hidden_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        
        queries_states = self.w_q(hidden_states) # (bsz, L, n_heads, head_dim)
        keys_states = self.w_k(hidden_states) # (bsz, L, n_heads, head_dim)
        values_states = self.w_v(hidden_states) # (bsz, L, n_heads, head_dim)
        
        queries_states = queries_states.transpose(1, 2) # (bsz, n_heads, L, head_dim)
        keys_states = keys_states.transpose(1, 2) # (bsz, n_heads, L, head_dim)
        values_states = values_states.transpose(1, 2) # (bsz, n_heads, L, head_dim)
        
        scores = torch.matmul(queries_states, keys_states.transpose(-2, -1)) * self.scaling_factor
        
        
        if attention_mask is not None:
            scores_full = scores + attention_mask
        else:
            scores_full = scores.clone()
            
        if sliding_window_mask is not None:
            scores_window = scores + sliding_window_mask
        else:
            scores_window = scores.clone()
            
        scores_full = torch.softmax(scores_full, dim=-1) @ values_states
        scores_window = torch.softmax(scores_window, dim=-1) @ values_states
        
        alpha = torch.sigmoid(self.w_g).view(1, self.num_heads, 1, 1)
        
        output = alpha * scores_full + (1 - alpha) * scores_window # (bsz, n_heads, L, head_dim)
        
        output = self.w_o(output.contiguous().view(bsz, seq_len, self.hidden_size))
        
        if self.training:
            output = torch.dropout(output, p=self.dropout, train=self.training)
            
        return output
