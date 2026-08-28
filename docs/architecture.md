# Eshmun architecture

Eshmun is a decoder-only, GPT-style causal protein LM, architecturally identical
to HuggingFace's OPT (`transformers.models.opt`) — same config fields and
state-dict key layout. This was verified directly, not assumed: loading a
`facebook/opt-*` checkpoint straight into `EshmunForCausalLM` and comparing
logits matched exactly. Only `model_type` differs (`"eshmun"` vs `"opt"`), so
`AutoModel` won't resolve one as the other automatically, but any OPT-derived
checkpoint — `facebook/opt-*`, or `hicai-zju/InstructProtein` (InstructProtein,
the model this project's whole training pipeline is built around) — loads in
directly via `EshmunForCausalLM.load_state_dict(...)`.

Source: `src/eshmun/models/eshmun/` (`configuration.py`, `modeling.py`).

## Why OPT-compatible, not a novel architecture

The point of this project isn't a new architecture — it's rationale-augmented
training (see [`kothar-pipeline.md`](kothar-pipeline.md)) on top of an existing,
published protein-instruction model (InstructProtein, ACL 2024,
arXiv:2310.03269). Being OPT-identical means InstructProtein's own weights load
in unmodified, its tokenizer and dedicated per-residue vocabulary (`ƤA`..`ƤY`)
carry over as-is, and any architectural difference observed downstream can be
attributed to training, not to a confound from re-implementing the base model.

## Config (`EshmunConfig`)

Defaults mirror OPT-125M-scale; the fields that matter for this project:

| field | default | notes |
|---|---|---|
| `vocab_size` | 50272 | InstructProtein/OPT tokenizer size; grows when `<think>`/`</think>` or other special tokens are added (see Kothar student, below) |
| `hidden_size` | 768 | |
| `num_hidden_layers` | 12 | Kothar's student subsets this to 6 (see below) |
| `ffn_dim` | 3072 | |
| `max_position_embeddings` | 2048 | matches InstructProtein's context length |
| `do_layer_norm_before` | `True` | pre-LN vs post-LN switch |
| `word_embed_proj_dim` | = `hidden_size` unless set smaller | adds projection layers when it differs (OPT's "embedding bottleneck" option) |
| `activation_function` | `"relu"` | OPT default, not GELU |
| `tie_word_embeddings` | `True` | LM head is tied to `embed_tokens` |
| `pad_token_id` / `bos_token_id` / `eos_token_id` | 1 / 2 / 2 | **`<bos>` and `<eos>` share id 2** — an OPT convention carried over as-is |

## Key classes (`modeling.py`)

- `EshmunLearnedPositionalEmbedding` — learned positional embeddings with a
  fixed `offset=2` (again, the OPT convention: position ids are shifted by 2 to
  make room for the shared bos/pad/eos ids at the low end of the embedding
  table).
- `EshmunAttention` — dispatches through `ALL_ATTENTION_FUNCTIONS.get_interface(...)`,
  i.e. backend-selectable (see below), not a fixed eager implementation.
- `EshmunDecoderLayer` — a `GradientCheckpointingLayer`, so gradient checkpointing
  is available for free via the standard HF training-arg switches.
- `EshmunPreTrainedModel` — sets `_supports_attention_backend`,
  `_supports_flash_attn`, `_supports_sdpa`, `_supports_flex_attn` all `True`:
  eager, Flash Attention, SDPA, and FlexAttention are all usable backends,
  selected the normal HF way (`attn_implementation=...`).
- `EshmunDecoder` — the transformer stack: embeddings → positional embeddings
  (offset 2) → N `EshmunDecoderLayer`s, with LayerDrop and an optional final
  layer norm (`do_layer_norm_before`/`_remove_final_layer_norm`), `DynamicCache`
  for KV caching during generation.
- `EshmunModel` — thin wrapper around `EshmunDecoder`.
- `EshmunForCausalLM` — adds the LM head (tied to `embed_tokens` when
  `tie_word_embeddings=True`) and inherits `GenerationMixin`, so the standard
  HF `.generate()` API (sampling, beam search, KV cache, `num_return_sequences`,
  etc.) works unmodified — used directly by
  `scripts/kothar/generate_sequences.py`.

## Tokenizer (`src/eshmun/tokenization.py`)

`EshmunTokenizer` is a thin `PreTrainedTokenizerFast` subclass that just fixes
Eshmun's default special tokens: `<bos>`, `<eos>`, `<unk>`, and pad aliased to
`<eos>` (no dedicated pad token — consistent with the OPT convention above).
It carries no protein-specific vocabulary of its own; in practice, this project
so far always loads InstructProtein's own tokenizer (or a copy of it with
`<think>`/`</think>` added — see the Kothar student), which already has the
dedicated per-residue `ƤX` tokens the data pipeline relies on (see
[`data-preparation.md`](data-preparation.md#residue-encoding)).

## Relationship to the tokenizer-ablation paper

A separate, already-submitted paper (Bioinformatics Advances, working dir
`~/work/phd/writing/protein_vocabulary`) found that for *causal* protein LMs,
amino-acid-level tokenization matches BPE-level tokenization in representation
quality once two evaluation confounds (fixed token-budget truncation,
last-token pooling) are controlled — tested at a fixed 127.46M non-embedding
parameter budget across AA/10K/20K/32K BPE vocabularies. That result is why
this project's earlier plan to extend InstructProtein's tokenizer with new
learned protein-BPE tokens was dropped in favor of just using InstructProtein's
existing (AA-level, `ƤX`-token) tokenizer as-is — see
[`kothar-pipeline.md`](kothar-pipeline.md#decision-history) for the pivot
timeline.

## An earlier, removed variant

An encoder-only "Eshmun-Zero" (protein *understanding*, not generation) existed
before the repo was reset to modeling-code-only; it is not present in the
current tree. If it resurfaces, `src/eshmun/models/zero/` is its natural
location, mirroring `src/eshmun/models/eshmun/`.
