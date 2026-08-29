# The Kothar pipeline

**Kothar** (Kothar-wa-Khasis / Chusor, Ugaritic/Phoenician god of craftsmanship
and skill — chosen 2026-08-21 to continue the `eshmun` mythology-naming
convention) is the codename for this project's current training pipeline and
the working title of its manuscript's subject model. Scripts live in
`scripts/kothar/`.

## Research question

Does inserting an explicit, programmatically-grounded chain-of-thought step
(`<think>...</think>`, citing real sequence/category features) before the
final answer improve instruction-following quality for a protein LM, compared
to training on the same `(instruction, answer)` pairs directly? The comparison
is three-way:

- **(a)** a compressed (layer-subsetted) InstructProtein student, trained
  **with** the `<think>` rationale block (`reasoning.py`'s
  `build_reasoning_block`).
- **(b)** the same student, trained on the same ground-truth
  `(instruction, answer)` pairs **without** the rationale block
  (`build_response_text` alone).
- **(c)** the original, untouched InstructProtein teacher, evaluated
  zero-shot — a baseline only, not fine-tuned.

## Decision history

The project's design went through two real pivots; both are load-bearing for
the manuscript's methods/discussion framing, so both are recorded here rather
than just left in git history.

**1. Tokenizer extension → dropped (decided 2026-08-21).** The original plan
was to extend InstructProtein's tokenizer with new BPE-merged protein tokens
(analogous to how `<think>`/`</think>` are added as single tokens now). This
was dropped because the tokenizer-ablation paper
(see [`architecture.md`](architecture.md#relationship-to-the-tokenizer-ablation-paper))
found AA-level tokenization already matches BPE-level quality for causal
protein LMs once evaluation confounds are controlled — undercutting the
motivation for a learned protein-BPE vocabulary. InstructProtein's *existing*
tokenizer (with its dedicated per-residue `ƤX` tokens) is used as-is instead.

**2. Teacher-output distillation → dropped in favor of programmatic ground
truth (decided 2026-08-21, reversed 2026-08-22).** The project briefly
considered having the teacher (InstructProtein) generate training data or
serve as a live KD/logit-matching signal — hence "distill InstructProtein" in
early framing. This was reverted: InstructProtein is now used **only as an
eval-time baseline** (condition c above); no stage of the pipeline uses a
teacher-supervised training signal (only the student's *initialization* is
teacher-derived, via layer subsetting — a compression step, not a training
signal). Two things drove the reversal:
  - **Cost/practicality**: the 1.3B teacher barely fits this machine's 6GB GPU
    even for short-sequence float32 inference (see build note in
    `probe_batch_size.py`'s use case), making live KD or bulk generation with
    it expensive and fragile.
  - **Quality**: the teacher-perplexity investigation below found the teacher
    has a weak grip on plain natural language — an argument that any teacher-
    generated instruction/rationale text risks being unreliable as a training
    signal, not just expensive to produce.
  This is also why the manuscript's title dropped "Distillation" (see
  `thinking_plm` memory) in favor of "Rationale-Augmented Fine-Tuning for a
  Compressed Protein Language Model" — the student's training targets are
  templated ground truth from `scripts/data/thinking/`, not teacher output.

## Stage 0 — Student construction (`build_student.py`, done 2026-08-21)

A 6-layer subset of the teacher (`hicai-zju/InstructProtein`, OPT-1.3B
architecture, 24 layers): teacher layers `[0, 1, 11, 12, 22, 23]` — 2 each
from the start, middle, end — copied verbatim via a filtered+reindexed state
dict (`select_layers`: first two, middle two straddling the center, last two).
Embeddings, positional embeddings, final layer norm, and the tied LM head are
copied as-is (unmodified, un-subsetted). `<think>`/`</think>` are then added
as special tokens and the student's embedding/LM head resized to match
(50287 → 50289 vocab).

Published (public): **huggingface.co/khairi/Kothar-student-seed-409M** — 409M
params. Round-trip verified: reload, tokenizer ids, and a finite-logits
sanity forward pass all checked out (`sanity_forward` in `build_student.py`).

This is the **warm start** for Stage 1 below — "continue the pretraining on
warm weights" was the explicit framing, not training from random init.

## Stage 1 — Continued pretraining (`pretrain.py`, in progress)

Domain-adaptive continued pretraining on a mixed corpus, before any
instruction-level tuning — the standard reason being: the student's layer
subsetting and the teacher's own weak grip on plain English (see the
diagnostic below) both argue for re-establishing solid next-token modeling,
across protein *and* general text, before layering task-specific SFT on top.

**Data** (`build_pretrain_mix.py` → pushed as `khairi/kothar-pretrain-mix-v1`
on the Hub): sampled from the same raw sources as an earlier, broader replay
mix (`khairi/uniref50-replay-mix-v1`, which also included a StarCoder code
split — deliberately dropped here, natural text only), at a fixed
**10:5:3:1 proteins:pubmed:finemath:fineweb-edu ratio**, anchored to 500,000
protein sequences:

| source | rows | of available | notes |
|---|---:|---:|---|
| UniRef50 | 500,000 | ~10,000,000 | `Ƥ`-per-residue encoded (see `data-preparation.md`) |
| PubMed abstracts | 250,000 | 283,302 | |
| FineMath | 150,000 | 54,615 | sampled **with replacement** (~2.75x duplication — not enough unique rows at this ratio/scale) |
| FineWeb-Edu | 50,000 | 257,645 | |

Total **950,000** rows, shuffled, seed 42. A disjoint held-out validation set
(`build_valid_holdout.py`) is built the same way at the same ratio, anchored
to 100 protein sequences, with entry-id exclusion enforced against the full
training mix (including FineMath's with-replacement duplicates) — used by
`eval_checkpoint.py`/`checkpoint_trend.py` (see
[`experiments.md`](experiments.md#stage-1-validation-perplexity-trend)).

**Training** (`pretrain.py`): `Trainer`-based, block-packed causal LM
(`group_texts`, 2048-token blocks, standard HF `run_clm.py`-style packing —
concatenate all tokenized examples, chop into contiguous blocks, drop the
final partial block). Float32 throughout, `fp16=False`/`bf16=False`
(project-wide convention — float16 caused unexpected errors previously).
Logs to Weights & Biases, project `Kothar`, run name `kothar-pretrain-409m`.

Effective config as actually run: `per_device_batch_size=24`,
`gradient_accumulation_steps=8`, single GPU → effective batch size 192;
learning rate 2e-5, cosine schedule (built for a 10,500-step total from the
run's first launch — see the note on `--max-steps` below), 500 warmup steps;
**1,050 steps/epoch** over the 950K-row mix at 2048-token blocks (201,447
packed blocks ÷ effective batch 192), 10 epochs requested → 10,500 total
steps planned. (An earlier version of this doc said "~6,296 steps/epoch" —
that number was actually from the botched first resume attempt that used the
wrong batch size, per-device 4 instead of 24; see [`status.md`](status.md).
1,050 is the correct figure for the run as actually configured.)

Stopping early, mid-run, must not use `--max-steps`: `pretrain.py` passes
`max_steps` straight into `TrainingArguments`, and `Trainer.create_scheduler()`
rebuilds the cosine LR schedule from `max_steps` (when set) *every time
training starts, including on resume* — so resuming with a smaller
`--max-steps` than the run was originally launched with would silently
recompute the cosine decay to finish by that new, smaller step count instead
of the original 10,500, distorting the LR trajectory for however much
training remains. To stop at a specific step without touching the schedule,
let the run continue unmodified and kill the process right after it writes
the checkpoint at that step (steps landing on the `save_steps` cadence, e.g.
any multiple of 250, always get a checkpoint) — see
[`status.md`](status.md) for the epoch-5 (step 5250) stopping point decided
2026-08-28.

A `SyncStateIntervalsCallback` forces `save_steps`/`logging_steps`/`eval_steps`
back to their CLI-requested values on every step — necessary because
`Trainer.train(resume_from_checkpoint=...)` otherwise silently restores
`save_steps` etc. from the checkpoint's `trainer_state.json`, ignoring a
CLI override meant to change them on resume (e.g. a faster save cadence after
a crash — see [`status.md`](status.md) for exactly this happening in
practice).

## Stage 2 — Thinking-aware SFT (not yet built)

Fine-tune the Stage-1 checkpoint on the `(instruction, reasoning, answer)`
triples from `scripts/data/thinking/` (see
[`data-preparation.md`](data-preparation.md)), once with the `<think>` block
included (condition a) and once without (condition b, same ground-truth
answers, `build_response_text` output only). No training code exists for this
yet — `src/eshmun/trainer/` was removed in the repo reset; this needs new
training code, not a resurrection of the old SFT/GRPO trainers.

## Evaluation

- **Stage 1**: per-source validation perplexity trend across checkpoints
  (`eval_checkpoint.py`, `checkpoint_trend.py`) — watching for a plateau
  before moving to Stage 2. See [`experiments.md`](experiments.md) for the
  numbers logged so far.
- **Generation quality (pilot, pretraining-stage only)**:
  `scripts/kothar/generate_sequences.py` samples sequences from a checkpoint;
  `scripts/eval/pretraining/plddt_esmfold2_forge.py` folds them (ESMFold2 via
  Biohub Forge) and reports mean pLDDT as a structural-plausibility proxy.
  This is a pretraining-stage sanity check, not the paper's main evaluation —
  the head-to-head (a) vs (b) vs (c) comparison happens after Stage 2.
