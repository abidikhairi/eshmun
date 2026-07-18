# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Eshmun is a family of Protein Language Models (PLMs) built in two paradigms:
- **Encoder** — Eshmun-Zero: protein understanding
- **Decoder (CLM)** — Eshmun (GPT-style): autoregressive protein sequence generation

All models are HuggingFace-compatible (`PreTrainedModel`, `PretrainedConfig`, standard outputs).

### Current research focus: thinking-aware instruction tuning

InstructProtein and ProLLaMA — two state-of-the-art protein instruction models — both train
directly on `(instruction, response)` pairs. Our hypothesis: inserting an explicit reasoning
step before the final answer — a chain of thought grounded in the sequence's biological
features — improves both protein **generation** (description/family → sequence) and
**annotation** (sequence → description/family) over direct instruction tuning.

Reasoning traces are built **programmatically** (template/rule-based) from known sequence
features — motifs, domains, family/superfamily membership, composition statistics — rather
than distilled from an external LLM or self-sampled (STaR-style). See `ROADMAP.md` for the
phased plan: baseline direct-SFT replication → thinking-aware dataset construction →
thinking-aware SFT → head-to-head evaluation.

The `<think>...</think>` response format this implies is already anticipated by the reward
registry in `trainer/grpo/reward_functions/format.py` (`PredicateReward`,
`RegexReward`) — usable now as format-compliance checks and later for RL-based refinement of
reasoning traces.

## Setup

- Python 3.13, managed via conda
- Install the package in editable mode: `pip install -e .`
- Type checker: `pyrefly check` (configured in `pyproject.toml`)

## Architecture

### Eshmun-Zero (`src/eshmun/models/zero/`)

Encoder-style model built on a pluggable attention registry (`build_attention`, selected via `config.attn_impl`): `mha`, `sliding_window`, `gqa`, `gated`, `qwen`, `token_filter`.

**Gated hybrid attention (`attention/gated.py`):**
- Computes a single set of attention scores, then reads them under two masks: a global (full-sequence) mask and a local sliding-window mask (`config.window_size`)
- Combined via a convex gate: `output = alpha * context_full + (1 - alpha) * context_window`, where `alpha = sigmoid(w_g)` is a learnable per-head scalar

**Key classes:**
- `EshmunZeroConfig` — `vocab_size`, `hidden_size`, `num_layers`, `attn_impl`, `window_size`, `use_rope`, etc.
- `EshmunZeroLayer` / `EshmunZeroDecoder` — pre-norm (RMSNorm) transformer stack: attention + MLP per layer
- `EshmunZero` — top-level `PreTrainedModel`; `lm_head` weights tied to `tokens_embed`; outputs `CausalLMOutputWithPast`

Attention masks are built in `EshmunZero`: `_build_3d_mask` → `(B,1,1,T)` additive full mask, `_build_sliding_window_mask` → `(B,1,T,T)` additive with window constraint.

### Eshmun (decoder) (`src/eshmun/models/eshmun/`)

Decoder-only model for autoregressive protein generation, architecturally similar to OPT.

**Key classes:**
- `EshmunConfig` — vocab_size=50272, supports `word_embed_proj_dim` ≠ `hidden_size` (adds projection layers)
- `EshmunDecoder` — transformer stack with learned positional embeddings (`offset=2`), LayerDrop, optional final layer norm, DynamicCache for KV
- `EshmunModel` — wraps `EshmunDecoder`
- `EshmunForCausalLM` — adds LM head (tied to embed_tokens), inherits `GenerationMixin`

Supports multiple attention backends: eager, Flash Attention, SDPA, FlexAttention (via `_supports_*` flags and `ALL_ATTENTION_FUNCTIONS`).

Pre/post layer norm is controlled by `do_layer_norm_before` in config.

## Type Checking

Pyrefly is the type checker. Suppress unavoidable errors with inline comments:
```python
# pyrefly: ignore [bad-override]
# pyrefly: ignore [bad-argument-type]
```

## Notebooks

- `notebooks/local/` — gitignored local experiments
