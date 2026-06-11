"""Unit tests for GroupedQueryAttention (eshmun.models.zero.attention.gqa)."""

import pytest
import torch

from eshmun.models.zero.configuration import EshmunZeroConfig
from eshmun.models.zero.attention.gqa import GroupedQueryAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(**overrides) -> EshmunZeroConfig:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=8,
        num_kv_heads=2,
        max_seq_len=128,
        dropout=0.0,
        use_rope=False,
    )
    defaults.update(overrides)
    return EshmunZeroConfig(**defaults)


def make_hidden(bsz: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    return torch.randn(bsz, seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Projection dimension tests
# ---------------------------------------------------------------------------

class TestProjectionDimensions:
    def test_query_projection_shape(self):
        config = make_config()
        attn = GroupedQueryAttention(config)
        assert attn.w_q.in_features == config.hidden_size
        assert attn.w_q.out_features == config.hidden_size

    def test_key_projection_shape(self):
        config = make_config()
        attn = GroupedQueryAttention(config)
        expected_out = attn.head_dim * config.num_kv_heads
        assert attn.w_k.in_features == config.hidden_size
        assert attn.w_k.out_features == expected_out

    def test_value_projection_shape(self):
        config = make_config()
        attn = GroupedQueryAttention(config)
        expected_out = attn.head_dim * config.num_kv_heads
        assert attn.w_v.in_features == config.hidden_size
        assert attn.w_v.out_features == expected_out

    def test_output_projection_shape(self):
        config = make_config()
        attn = GroupedQueryAttention(config)
        assert attn.w_o.in_features == config.hidden_size
        assert attn.w_o.out_features == config.hidden_size

    def test_kv_heads_smaller_than_q_heads(self):
        # num_kv_heads < num_attention_heads is the defining property of GQA
        config = make_config(num_attention_heads=8, num_kv_heads=2)
        attn = GroupedQueryAttention(config)
        kv_dim = attn.head_dim * config.num_kv_heads
        assert attn.w_k.out_features == kv_dim
        assert attn.w_v.out_features == kv_dim
        assert kv_dim < config.hidden_size


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_output_shape_basic(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, 16, config.hidden_size)

    def test_output_shape_batch_size_1(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(1, 10, config.hidden_size)
        out = attn(x)
        assert out.shape == (1, 10, config.hidden_size)

    def test_output_shape_large_batch(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(8, 32, config.hidden_size)
        out = attn(x)
        assert out.shape == (8, 32, config.hidden_size)

    def test_output_shape_seq_len_1(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 1, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, 1, config.hidden_size)

    def test_output_dtype_preserved(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# Attention mask tests
# ---------------------------------------------------------------------------

class TestAttentionMask:
    def test_forward_without_mask(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x, attention_mask=None)
        assert out.shape == (2, 16, config.hidden_size)

    def test_forward_with_causal_mask(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        bsz, seq_len = 2, 16
        x = make_hidden(bsz, seq_len, config.hidden_size)
        # additive causal mask: (bsz, 1, seq_len, seq_len) with -inf above diagonal
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
        out = attn(x, attention_mask=mask)
        assert out.shape == (bsz, seq_len, config.hidden_size)

    def test_mask_changes_output(self):
        # A fully masked sequence should produce a different output than no mask
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        bsz, seq_len = 1, 8
        x = make_hidden(bsz, seq_len, config.hidden_size)
        out_no_mask = attn(x, attention_mask=None)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        out_masked = attn(x, attention_mask=causal_mask)
        assert not torch.allclose(out_no_mask, out_masked)


# ---------------------------------------------------------------------------
# GQA-specific: KV head expansion
# ---------------------------------------------------------------------------

class TestKVHeadExpansion:
    def test_mha_mode_num_kv_equals_num_q(self):
        # When num_kv_heads == num_attention_heads, GQA reduces to MHA
        config = make_config(num_attention_heads=4, num_kv_heads=4)
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, 16, config.hidden_size)

    def test_mqa_mode_single_kv_head(self):
        # When num_kv_heads == 1, GQA reduces to Multi-Query Attention
        config = make_config(num_attention_heads=8, num_kv_heads=1)
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, 16, config.hidden_size)

    def test_groups_of_four(self):
        config = make_config(num_attention_heads=8, num_kv_heads=2)
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, 16, config.hidden_size)


# ---------------------------------------------------------------------------
# RoPE integration
# ---------------------------------------------------------------------------

class TestRoPEIntegration:
    def test_rope_disabled_by_default(self):
        config = make_config(use_rope=False)
        attn = GroupedQueryAttention(config)
        assert not hasattr(attn, "rope") or attn.rope is None

    def test_rope_enabled_creates_rope_module(self):
        config = make_config(use_rope=True)
        attn = GroupedQueryAttention(config)
        assert hasattr(attn, "rope") and attn.rope is not None

    def test_output_shape_with_rope(self):
        config = make_config(use_rope=True)
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, 16, config.hidden_size)

    def test_rope_output_differs_from_no_rope(self):
        torch.manual_seed(0)
        x = make_hidden(2, 16, 64)
        # same random seed → same weights, only RoPE differs
        torch.manual_seed(42)
        attn_no_rope = GroupedQueryAttention(make_config(use_rope=False)).eval()
        torch.manual_seed(42)
        attn_rope = GroupedQueryAttention(make_config(use_rope=True)).eval()
        out_no_rope = attn_no_rope(x)
        out_rope = attn_rope(x)
        assert not torch.allclose(out_no_rope, out_rope)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output_eval(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x = make_hidden(2, 16, config.hidden_size)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        config = make_config()
        attn = GroupedQueryAttention(config).eval()
        x1 = make_hidden(2, 16, config.hidden_size)
        x2 = make_hidden(2, 16, config.hidden_size)
        out1 = attn(x1)
        out2 = attn(x2)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_to_all_projections(self):
        config = make_config()
        attn = GroupedQueryAttention(config)
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradients_flow_with_rope(self):
        config = make_config(use_rope=True)
        attn = GroupedQueryAttention(config)
        x = make_hidden(2, 16, config.hidden_size)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
