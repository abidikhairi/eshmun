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

        # Project the full hidden vector, then split into heads, so the heads are
        # independent (untied).
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

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

        queries_states = (
            self.w_q(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, n_heads, L, head_dim)
        keys_states = (
            self.w_k(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, n_heads, L, head_dim)
        values_states = (
            self.w_v(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, n_heads, L, head_dim)

        if self.use_rope:
            queries_states, keys_states = self.rope(queries_states, keys_states)

        scores = (
            torch.matmul(queries_states, keys_states.transpose(-2, -1))
            * self.scaling_factor
        )

        # Two attention views over the same scores: a global (full) view and a
        # local (sliding-window) view. softmax is not in-place, so both can read
        # the shared `scores` tensor directly.
        scores_full = scores if attention_mask is None else scores + attention_mask
        scores_window = scores if sliding_window_mask is None else scores + sliding_window_mask

        context_full = torch.matmul(torch.softmax(scores_full, dim=-1), values_states)
        context_window = torch.matmul(torch.softmax(scores_window, dim=-1), values_states)

        alpha = torch.sigmoid(self.w_g).view(1, self.num_heads, 1, 1)

        output = alpha * context_full + (1 - alpha) * context_window  # (bsz, n_heads, L, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.w_o(output)

        if self.training:
            output = torch.dropout(output, p=self.dropout, train=self.training)

        return output
