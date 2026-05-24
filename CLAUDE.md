# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Eshmun is a family of Protein Language Models (PLMs) built in two paradigms:
- **Encoder (MLM)** — Eshmun-Zero: protein understanding via masked language modeling
- **Decoder (CLM)** — Eshmun (GPT-style): autoregressive protein sequence generation

All models are HuggingFace-compatible (`PreTrainedModel`, `PretrainedConfig`, standard outputs).

## Setup

- Python 3.13, managed via conda
- Install the package in editable mode: `pip install -e .`
- Type checker: `pyrefly check` (configured in `pyproject.toml`)

## Architecture

### Eshmun-Zero (`src/eshmun/models/zero/`)

Encoder-only model trained with masked language modeling on protein sequences.

**Key design — gated hybrid attention (`EshmunZeroGatedAttention`):**
- Each layer runs two independent attention heads in parallel: local sliding-window (`local_window_size` tokens) and global full-sequence attention
- Combined via a convex gate: `output = alpha * local + (1 - alpha) * global`, where `alpha = sigmoid(s)` and `s` is a per-layer learnable scalar
- Alpha values can be inspected at runtime with `output_alphas=True`

**Dimension flow:**
```
input_ids → EshmunZeroEmbeddings (embedding_size=4096)
          → token_to_hidden linear (→ hidden_size=768)
          → EshmunZeroEncoder (N × [GatedAttention + FFN])
          → hidden_to_token linear (→ embedding_size=4096)
          → lm_head (→ vocab_size=25)
```

The embedding/hidden size split allows a large embedding space (amino acid vocabulary) independent of the transformer width. LM head weights are tied to the input embedding.

**Key classes:**
- `EshmunZeroConfig` — vocab_size=25 (amino acid alphabet), local_window_size=12, alpha_init=0.0 (starts at alpha=0.5)
- `EshmunZeroModel` — base encoder (no LM head), outputs `BaseModelOutputWithPooling`
- `EshmunZeroForMaskedLM` — adds MLM head with tied embeddings, outputs `MaskedLMOutput`
- `EshmunPooler` — pools `[CLS]` token for sequence-level tasks

Attention masks are built in `models.py`: `_prepare_global_attention_mask` → `(B,1,1,T)` additive, `_prepare_local_attention_mask` → `(B,1,T,T)` additive with window constraint.

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

- `notebooks/eshmun_zero/` — MLM training, model testing, HuggingFace Hub push
- `notebooks/zero_test/` — exploratory testing
- `notebooks/local/` — gitignored local experiments
