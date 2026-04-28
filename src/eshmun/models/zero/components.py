"""
Shared transformer building blocks for Eshmun-Zero.

Components:
  - EshmunEmbeddings    : token + position + token-type embeddings
  - EshmunFeedForward   : position-wise feed-forward (FFN) with LayerNorm
  - EshmunLayer         : one full transformer layer (GatedAttention + FFN)
  - EshmunEncoder       : stack of EshmunLayer (for MLM / encoder)
  - EshmunPooler        : [CLS] pooler for sequence classification
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eshmun.models.zero.attention import EshmunGatedAttention
from eshmun.models.zero.configuration import EshmunZeroConfig

ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "tanh": torch.tanh,
}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EshmunEmbeddings(nn.Module):
    """
    Word + absolute position + token-type embeddings, followed by LayerNorm
    and dropout.
    """

    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        self.wte = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape

        if position_ids is None:
            position_ids = self.position_ids[:, :T]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                B, T, dtype=torch.long, device=input_ids.device
            )

        word_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)
        tt_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + tt_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------


class EshmunFeedForward(nn.Module):
    """
    Position-wise feed-forward:  FFN(x) = LayerNorm( x + W2 act(W1 x) )
    """

    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN.get(config.hidden_act, F.gelu)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(residual + hidden_states)


# ---------------------------------------------------------------------------
# Single Transformer Layer
# ---------------------------------------------------------------------------


class EshmunLayer(nn.Module):
    """
    One Eshmun-Zero transformer layer:

        x' = GatedAttention(x)     # local + global, gated by alpha
        y  = FeedForward(x')

    Args:
        config: EshmunZeroConfig
        causal: whether to use causal masking in the attention branches
    """

    def __init__(self, config: EshmunZeroConfig, causal: bool = False):
        super().__init__()
        self.attention = EshmunGatedAttention(config, causal=causal)
        self.ffn = EshmunFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden_states: (B, T, D)
            alpha: current gate value (scalar tensor)
        """
        hidden_states, alpha = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.ffn(hidden_states)
        return hidden_states, alpha


# ---------------------------------------------------------------------------
# Encoder (bidirectional — for MLM)
# ---------------------------------------------------------------------------


class EshmunEncoder(nn.Module):
    """
    Stack of EshmunLayer with causal=False (bidirectional).
    Used by EshmunModel (base) and EshmunForMaskedLM.
    """

    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [EshmunLayer(config, causal=False) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_alphas: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, T, D)
            attention_mask: (B, 1, 1, T) additive mask, optional
            output_alphas: if True, return per-layer alpha values

        Returns:
            hidden_states: (B, T, D)
            alphas: list of scalar tensors (one per layer) if output_alphas else None
        """
        alphas = [] if output_alphas else None

        for layer in self.layers:
            hidden_states, alpha = layer(hidden_states, attention_mask=attention_mask)
            if output_alphas:
                alphas.append(alpha)

        return hidden_states, alphas


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------


class EshmunPooler(nn.Module):
    """
    Pools the [CLS] token representation for sequence-level tasks.
    """

    def __init__(self, config: EshmunZeroConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0, :]
        return self.activation(self.dense(cls_token))
