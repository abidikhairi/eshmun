# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Eshmun is a Protein Language Model (PLM): a decoder-only, GPT-style model
(`src/eshmun/models/eshmun/`) for autoregressive protein sequence generation.
HuggingFace-compatible (`PreTrainedModel`, `PretrainedConfig`, standard outputs).

An earlier encoder variant (Eshmun-Zero, protein understanding) has been removed from the
codebase; if it resurfaces it'll be under `src/eshmun/models/zero/` again.

## Status

The repository was reset to its core: modeling code and data preparation scripts only.
Prior training infrastructure (SFT/GRPO/distillation trainers), evaluation pipelines,
notebooks, and docs from the earlier thinking-aware instruction-tuning pilot were removed —
that work restarted under the codename **Kothar**: a layer-subsetted (409M) InstructProtein
student, continued-pretrained (Stage 1, in progress) then fine-tuned with vs. without a
programmatic `<think>...</think>` rationale block (Stage 2, not yet built), compared
head-to-head against each other and against the zero-shot teacher. See
[`docs/kothar-pipeline.md`](docs/kothar-pipeline.md) for the full plan and decision history,
[`docs/status.md`](docs/status.md) for what's running right now, and `ROADMAP.md` for the
phase checklist. `docs/` doubles as the working notes for the accompanying manuscript
(`~/work/phd/writing/thinking_plm`) — check it before assuming any research conclusion
still holds, and update it (not just code) when a design decision changes.

## Setup

- Python 3.13, managed via conda
- Install the package in editable mode: `pip install -e .`
- Type checker: `pyrefly check` (configured in `pyproject.toml`)

## Architecture

### Eshmun (decoder) (`src/eshmun/models/eshmun/`)

Decoder-only model for autoregressive protein generation, architecturally identical to
HuggingFace's OPT (`transformers.models.opt`) — same config fields and state-dict key
layout (`model.decoder.embed_tokens`, `model.decoder.layers.N.self_attn.{q,k,v,out}_proj`,
`lm_head.weight`, ...), verified by loading a `facebook/opt-*` checkpoint directly into
`EshmunForCausalLM` and matching logits exactly. Only `model_type` differs (`"eshmun"` vs
`"opt"`), so `AutoModel` won't auto-resolve between them, but a `facebook/opt-*` or any
OPT-derived checkpoint (e.g. `hicai-zju/InstructProtein`) can be loaded in directly via
`EshmunForCausalLM.load_state_dict(...)`.

**Key classes:**
- `EshmunConfig` — vocab_size=50272, supports `word_embed_proj_dim` ≠ `hidden_size` (adds projection layers)
- `EshmunDecoder` — transformer stack with learned positional embeddings (`offset=2`), LayerDrop, optional final layer norm, DynamicCache for KV
- `EshmunModel` — wraps `EshmunDecoder`
- `EshmunForCausalLM` — adds LM head (tied to embed_tokens), inherits `GenerationMixin`

Supports multiple attention backends: eager, Flash Attention, SDPA, FlexAttention (via `_supports_*` flags and `ALL_ATTENTION_FUNCTIONS`).

Pre/post layer norm is controlled by `do_layer_norm_before` in config.

### Tokenizer (`src/eshmun/tokenization.py`)

`EshmunTokenizer` — thin `PreTrainedTokenizerFast` wrapper with Eshmun's default special
tokens (`<bos>`, `<eos>`, `<unk>`, pad aliased to `<eos>`).

### Data preparation (`scripts/data/thinking/`)

Standalone scripts (FASTA parsing, UniProt field extraction, SCOP/PPI dataset construction,
instruction-pool generation, KG building, identity-based splitting) for building
instruction/annotation datasets from UniProt/SCOP sources. Not wired to any trainer
currently — reusable building blocks for whatever training pipeline comes next. Details:
[`docs/data-preparation.md`](docs/data-preparation.md).

### Kothar pipeline (`scripts/kothar/`, `scripts/eval/pretraining/`)

Student construction, Stage-1 continued-pretraining data/training scripts, checkpoint
evaluation (validation perplexity, generation sampling, pLDDT). Details, current numbers,
and live training status: [`docs/kothar-pipeline.md`](docs/kothar-pipeline.md),
[`docs/experiments.md`](docs/experiments.md), [`docs/status.md`](docs/status.md).

## Conventions

- **Always load/train models in float32**, never float16 (`dtype=torch.float32`,
  `fp16=False`/`bf16=False` in `TrainingArguments`) — float16 has caused unexpected errors
  in this project before. Don't conditionally switch to float16 based on CUDA availability.

## Type Checking

Pyrefly is the type checker. Suppress unavoidable errors with inline comments:
```python
# pyrefly: ignore [bad-override]
# pyrefly: ignore [bad-argument-type]
```
