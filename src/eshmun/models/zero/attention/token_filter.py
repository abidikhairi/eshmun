import torch
from torch import nn

from eshmun.models.zero.attention import BaseAttentionModule
from eshmun.models.zero.configuration import EshmunZeroConfig
from eshmun.models.zero.attention.rope import RoPE


class TokenFilterAttention(BaseAttentionModule):
    def __init__(self, config: EshmunZeroConfig):
        super().__init__(config)

        if self.use_rope:
            self.rope = RoPE(self.head_dim, self.config.max_seq_len, base=10_000)

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.w_gate = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self._init_params()
        
    def _init_params(self):
        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, 2.0) # sigmoid(2.0) -> 0.88

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        bsz, seq_len, _ = hidden_states.shape

        # hidden_states = hidden_states.view(bsz, seq_len)

        gate = torch.sigmoid(self.w_gate(hidden_states))  # (bsz, num_heads, seq_len, 1)
        gate = gate.transpose(1, 2).unsqueeze(-1)

        queries_states = (
            self.w_q(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, num_heads, seq_len, head_dim)

        keys_states = (
            self.w_k(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, num_heads, seq_len, head_dim)

        values_states = (
            self.w_v(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, seq_len, num_heads, head_dim)

        if self.use_rope:
            queries_states, keys_states = self.rope(queries_states, keys_states)

        queries_states = queries_states * gate
        keys_states = keys_states.transpose(-2, -1)

        scores = torch.matmul(queries_states, keys_states) * self.scaling_factor

        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.softmax(scores, dim=-1)

        output = torch.matmul(scores, values_states)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        output = self.w_o(output)

        return output
