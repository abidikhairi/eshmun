import torch
from torch import nn

from eshmun.models.gpt.gpt_configuration import EshmunGPTConfig
from eshmun.models.gpt.attention.base import BaseAttentionModule
from eshmun.models.gpt.attention.rope import RoPE


class MultiHeadAttention(BaseAttentionModule):
    def __init__(self, config: EshmunGPTConfig):
        super().__init__(config)
        
        if self.use_rope:
            self.rope = RoPE(self.head_dim, self.config.max_seq_len, base=10_000)
        
        self.w_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.w_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.w_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        bsz, seq_len, _ = hidden_states.shape
        
        hidden_states = hidden_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        
        query = self.w_q(hidden_states).transpose(1, 2) # (bsz, num_heads, seq_len, head_dim)
        key = self.w_k(hidden_states).transpose(1, 2) # (bsz, num_heads, seq_len, head_dim)
        value = self.w_v(hidden_states).transpose(1, 2) # (bsz, seq_len, num_heads, head_dim)
        
        if self.use_rope:
            query, key = self.rope(query, key)
        
        key = key.transpose(-2, -1)
        
        scores = torch.matmul(query, key) * self.scaling_factor
        
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.softmax(scores, dim=-1)
        
        output = self.w_o(
            torch.matmul(scores, value).contiguous().view(bsz, seq_len, -1)
        )
        
        if self.training:
            output = torch.dropout(output, p=self.dropout, train=self.training)
        
        return output
