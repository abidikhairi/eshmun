"""
Attention components for Eshmun-Zero.

Implements:
  - EshmunZeroSelfAttention    : standard multihead self-attention
  - EshmunGatedAttention       : parallel combination via learnable scalar alpha,
                                  with beta = 1 - alpha (single-parameter convex gate)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eshmun.models.zero.configuration import EshmunZeroConfig


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class EshmunZeroSelfAttention(nn.Module):
    """
    Standard multi-head scaled dot-product attention.

    F(X) = softmax( Q_g K_g^T / sqrt(d_k) ) V_g
    """

    def __init__(self, config: EshmunZeroConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})."
            )

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, d_k)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) -> (B, T, D)"""
        B, H, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)
            attention_mask: (B, 1, 1, T) additive mask (0 or -inf), optional

        Returns:
            output: (B, T, D)
        """
        Q = self._split_heads(self.q(hidden_states))  # (B, H, T, d_k)
        K = self._split_heads(self.k(hidden_states))
        V = self._split_heads(self.v(hidden_states))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, T, d_k)
        context = self._merge_heads(context)  # (B, T, D)

        return self.out_proj(context)



# ---------------------------------------------------------------------------
# Gated Parallel Attention (Eshmun-Zero core layer)
# ---------------------------------------------------------------------------


class EshmunZeroGatedAttention(nn.Module):
    """
    Parallel local + global attention combined via a single learnable gate.

    F(X) = LayerNorm( X + alpha * A_local(X) + (1 - alpha) * A_global(X) )

    where alpha = sigmoid(s), s is a scalar parameter learned per layer.

    This is a convex combination: alpha in (0, 1), alpha + beta = 1.

    Args:
        config: EshmunConfig
        causal: if True, both attention branches apply causal masking
                (used for causal language modelling).
    """

    def __init__(self, config: EshmunZeroConfig, causal: bool = False):
        super().__init__()
        self.causal = causal

        self.local_attn = EshmunZeroSelfAttention(config)
        self.global_attn = EshmunZeroSelfAttention(config)

        # Learnable raw scalar; alpha = sigmoid(s)
        self.gate_s = nn.Parameter(torch.tensor(config.alpha_init))

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    @property
    def alpha(self) -> torch.Tensor:
        """Convex gate weight for local attention, in (0, 1)."""
        return torch.sigmoid(self.gate_s)

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_attention_mask: Optional[torch.Tensor] = None,
        local_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, D)
            global_attention_mask: (B, 1, 1, T) additive mask for global attention, optional
            local_attention_mask: (B, 1, T, T) additive mask for local window attention, optional

        Returns:
            output: (B, T, D)  — layer-normed residual output
            alpha:  scalar tensor — current gate value (for inspection / logging)
        """
        a = self.alpha

        local_out = self.local_attn(
            hidden_states, attention_mask=local_attention_mask
        )

        global_out = self.global_attn(
            hidden_states, attention_mask=global_attention_mask
        )

        combined = a * local_out + (1.0 - a) * global_out
        combined = self.dropout(combined)

        output = self.layer_norm(hidden_states + combined)

        return output, a
