import torch
from torch import nn
from eshmun.models.zero.configuration import EshmunZeroConfig
from eshmun.models.zero.attention.base import BaseAttentionModule
from eshmun.models.zero.attention.rope import RoPE


class GroupedQueryAttention(BaseAttentionModule):
    def __init__(self, config: EshmunZeroConfig):
        super().__init__(config)
        self.num_kv_heads = config.num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0, """
            num_attention_heads ({}) must be divisible by num_kv_heads ({})""".format(
            self.num_heads, self.num_kv_heads
        )

        if self.use_rope:
            self.rope = RoPE(self.head_dim, self.config.max_seq_len, base=10_000)

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.w_k = nn.Linear(
            self.hidden_size, self.head_dim * self.num_kv_heads, bias=False
        )

        self.w_v = nn.Linear(
            self.hidden_size, self.head_dim * self.num_kv_heads, bias=False
        )

        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:

        bsz, seq_len, _ = hidden_states.shape

        queries_states = self.w_q(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )  # (bsz, L, n_heads, head_dim)

        keys_states = self.w_k(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )  # (bsz, L, n_kv_heads, head_dim)
        values_states = self.w_v(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )  # (bsz, L, n_kv_heads, head_dim)

        keys_states = keys_states.repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=2
        )  # (bsz, L, n_heads, head_dim)
        values_states = values_states.repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=2
        )  # (bsz, L, n_heads, head_dim)

        queries_states = queries_states.transpose(1, 2)  # (bsz, n_heads, L, head_dim)
        keys_states = keys_states.transpose(1, 2)  # (bsz, n_heads, L, head_dim)
        values_states = values_states.transpose(1, 2)  # (bsz, n_heads, L, head_dim)

        if self.use_rope:
            queries_states, keys_states = self.rope(queries_states, keys_states)

        scores = (
            torch.matmul(queries_states, keys_states.transpose(-2, -1))
            * self.scaling_factor
        )

        if attention_mask is not None:
            scores = scores + attention_mask

        scores = torch.softmax(scores, dim=-1)

        output = (
            torch.matmul(scores, values_states)
            .contiguous()
            .view(bsz, seq_len, self.hidden_size)
        )

        output = self.w_o(output)

        if self.training:
            output = torch.dropout(output, p=self.dropout, train=self.training)

        return output
