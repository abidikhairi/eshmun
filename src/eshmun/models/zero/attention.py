"""
Attention components for Eshmun-Zero.

Implements:
  - EshmunGlobalSelfAttention  : standard scaled dot-product attention (full sequence)
  - EshmunLocalSelfAttention   : windowed self-attention with fixed window size w
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
# Helpers
# ---------------------------------------------------------------------------


def _make_causal_mask(
    seq_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Upper-triangular mask that blocks future tokens (additive, -inf style)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (seq_len, seq_len)


# ---------------------------------------------------------------------------
# Global Self-Attention
# ---------------------------------------------------------------------------


class EshmunGlobalSelfAttention(nn.Module):
    """
    Standard multi-head scaled dot-product attention over the full sequence.

    Complexity: O(n^2 * d)

    F_global(X) = softmax( Q_g K_g^T / sqrt(d_k) ) V_g
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
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)
            attention_mask: (B, 1, 1, T) additive mask (0 or -inf), optional
            causal: if True, apply causal (upper-triangular) masking

        Returns:
            output: (B, T, D)
        """
        Q = self._split_heads(self.q(hidden_states))  # (B, H, T, d_k)
        K = self._split_heads(self.k(hidden_states))
        V = self._split_heads(self.v(hidden_states))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if causal:
            T = hidden_states.size(1)
            causal_mask = _make_causal_mask(
                T, device=hidden_states.device, dtype=scores.dtype
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, T, d_k)
        context = self._merge_heads(context)  # (B, T, D)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Local Self-Attention (fixed window)
# ---------------------------------------------------------------------------


class EshmunLocalSelfAttention(nn.Module):
    """
    Multi-head local self-attention with a fixed symmetric window of size w.

    Token i attends only to tokens in [i - w//2, i + w//2] (clipped at boundaries).

    Complexity: O(n * w * d)

    Implementation uses unfolding to gather local key/value windows efficiently,
    avoiding explicit loops over positions.
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
        self.window_size = config.local_window_size

        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)
            attention_mask: (B, 1, 1, T) additive mask (0 or -inf), optional.
                            Applied to the gathered window scores.
            causal: if True, tokens cannot attend to future positions even
                    within the local window.

        Returns:
            output: (B, T, D)
        """
        B, T, D = hidden_states.shape
        w = self.window_size
        half_w = w // 2
        H = self.num_heads
        d_k = self.head_dim

        Q = self.q(hidden_states)  # (B, T, D)
        K = self.k(hidden_states)
        V = self.v(hidden_states)

        # Reshape to (B, H, T, d_k)
        def split_heads(x):
            return x.view(B, T, H, d_k).transpose(1, 2)

        Q = split_heads(Q)  # (B, H, T, d_k)
        K = split_heads(K)
        V = split_heads(V)

        # Pad K and V along the time dimension so every token has a full window
        # Padding: half_w on each side with -inf (K) / 0 (V)
        # Shape after pad: (B, H, T + 2*half_w, d_k)
        pad = (0, 0, half_w, half_w)  # (last_dim_left, last_dim_right, T_left, T_right)
        K_pad = F.pad(K, pad, value=0.0)
        V_pad = F.pad(V, pad, value=0.0)

        # Build a mask for padded positions so they get -inf attention score
        # True = valid, False = padding
        valid = torch.ones(B, T, device=hidden_states.device, dtype=torch.bool)
        valid_pad = F.pad(valid, (half_w, half_w), value=False)  # (B, T + 2*half_w)

        # Gather windows: for each position i, collect indices [i, i+w)
        # Result: (B, H, T, w, d_k)
        idx = torch.arange(T, device=hidden_states.device).unsqueeze(1) + torch.arange(
            w, device=hidden_states.device
        ).unsqueeze(
            0
        )  # (T, w)

        # Expand idx for batch and heads
        idx_K = idx.unsqueeze(0).unsqueeze(0).expand(B, H, T, w)  # (B, H, T, w)
        idx_V = idx_K

        K_win = K_pad.gather(
            2, idx_K.reshape(B, H, T * w, 1).expand(B, H, T * w, d_k)
        ).view(
            B, H, T, w, d_k
        )  # (B, H, T, w, d_k)

        V_win = V_pad.gather(
            2, idx_V.reshape(B, H, T * w, 1).expand(B, H, T * w, d_k)
        ).view(B, H, T, w, d_k)

        # Compute local attention scores: (B, H, T, 1, d_k) x (B, H, T, d_k, w)
        Q_exp = Q.unsqueeze(3)  # (B, H, T, 1, d_k)
        scores = torch.matmul(Q_exp, K_win.transpose(-2, -1))  # (B, H, T, 1, w)
        scores = scores.squeeze(3) / self.scale  # (B, H, T, w)

        # Mask padded positions in the window
        valid_win = valid_pad[:, idx].unsqueeze(1).expand(B, H, T, w)  # (B, H, T, w)
        scores = scores.masked_fill(~valid_win, float("-inf"))

        # Causal masking within the window
        if causal:
            # Position j in the window corresponds to absolute index i - half_w + j
            # Mask j if (i - half_w + j) > i, i.e. j > half_w
            j_idx = torch.arange(w, device=hidden_states.device)  # (w,)
            future_mask = j_idx.unsqueeze(0) > half_w  # (1, w) True = future
            scores = scores.masked_fill(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Optional external attention mask: (B, 1, 1, T) -> broadcast over window
        # We apply the mask for the query positions (not the window dimension).
        if attention_mask is not None:
            # attention_mask shape: (B, 1, 1, T) — squeeze to (B, T)
            query_mask = attention_mask.squeeze(1).squeeze(1)  # (B, T)
            scores = scores + query_mask.unsqueeze(1).unsqueeze(-1)

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, w)
        # Replace NaN from all-inf rows (fully masked tokens) with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (B, H, T, w) x (B, H, T, w, d_k) -> (B, H, T, d_k)
        context = torch.matmul(attn_weights.unsqueeze(3), V_win)  # (B, H, T, 1, d_k)
        context = context.squeeze(3)  # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) -> (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Gated Parallel Attention (Eshmun-Zero core layer)
# ---------------------------------------------------------------------------


class EshmunGatedAttention(nn.Module):
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

        self.local_attn = EshmunLocalSelfAttention(config)
        self.global_attn = EshmunGlobalSelfAttention(config)

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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, D)
            attention_mask: (B, 1, 1, T) additive mask, optional

        Returns:
            output: (B, T, D)  — layer-normed residual output
            alpha:  scalar tensor — current gate value (for inspection / logging)
        """
        a = self.alpha

        local_out = self.local_attn(
            hidden_states, attention_mask=attention_mask, causal=self.causal
        )

        global_out = self.global_attn(
            hidden_states, attention_mask=attention_mask, causal=self.causal
        )

        combined = a * local_out + (1.0 - a) * global_out
        combined = self.dropout(combined)

        output = self.layer_norm(hidden_states + combined)
        return output, a
