# Eshmun Roadmap

> Thinking-aware instruction tuning for protein generation and annotation —
> codenamed **Kothar**. Full narrative and decision history:
> [`docs/kothar-pipeline.md`](docs/kothar-pipeline.md); live status:
> [`docs/status.md`](docs/status.md).

**Horizon:** Short-term · 2–4 months
**Status:** Stage 1 (continued pretraining) in progress
**Last updated:** 2026-08-28

---

## Hypothesis

InstructProtein and ProLLaMA — two state-of-the-art protein instruction models —
both train directly on `(instruction, response)` pairs. We hypothesize that
inserting an explicit chain-of-thought step before the final answer, grounded
in the sequence's biological features, improves both:

- **Generation** — text/family description → protein sequence
- **Annotation** — protein sequence → functional/family description

over direct instruction tuning, without requiring an external teacher model or
RL-based self-sampling to produce the reasoning traces.

## Approach

Reasoning traces are constructed **programmatically** from known sequence
features (motifs, domains, superfamily/family membership, composition
statistics, length, etc.) rather than distilled from a stronger LLM or
self-taught (STaR). This keeps dataset construction fully controllable and
reproducible, at the cost of being limited to reasoning patterns that are
programmatically expressible. Reference model: `hicai-zju/InstructProtein`,
used as an **eval-only baseline** — no stage of the pipeline uses a
teacher-supervised training signal (see `docs/kothar-pipeline.md`'s decision
history for why this was reconsidered and settled).

---

## Phase 0 — Student construction — **done** (2026-08-21)

- [x] Build a 6-layer, layer-subsetted student from the teacher
      (`scripts/kothar/build_student.py`) — published at
      `khairi/Kothar-student-seed-409M`
- [x] Add `<think>`/`</think>` special tokens, verify with a round-trip +
      finite-logits sanity check

## Phase 1 — Stage 1: continued pretraining — **in progress**

- [x] Build the protein + natural-text replay mix
      (`scripts/kothar/build_pretrain_mix.py` → `khairi/kothar-pretrain-mix-v1`,
      950K rows, 10:5:3:1 protein:pubmed:finemath:fineweb-edu)
- [x] Build a disjoint held-out validation set (`build_valid_holdout.py`)
- [ ] Warm-start the student and continue-pretrain to a validation-perplexity
      plateau (`scripts/kothar/pretrain.py`) — see `docs/status.md` for the
      live checkpoint/step and `docs/experiments.md` for the trend so far
      (not yet plateaued as of step ~2700/10500)
- [ ] Decide the stopping point (plateau reached, or a deliberate early stop)
      and record it in `docs/status.md`

## Phase 2 — Thinking-aware dataset construction — **mostly done**

- [x] Define the reasoning-trace schema per task (`scripts/data/thinking/reasoning.py`:
      annotation cites the entry's other real facts; generation cites
      category-level aggregate stats, not instance-specific facts, to avoid
      causal inversion/leakage)
- [x] Implement KG construction from UniProt (human annotation + generation)
      and SCOP/PPI (all-organism) sources
      (`build_annotation_kg.py`, `build_scop_kg.py`, `build_ppi_dataset.py`, ...)
- [x] Implement the `<think>...</think>` templating layer (`build_reasoning_block`)
- [x] Build the `(instruction, reasoning, answer)` triples for task family A
      (`build_task_a_annotation.py`, `build_task_a_generation.py`) and task
      family B (SCOP/PPI)
- [x] Identity-based train/val/test split via MMseqs2, guarding against
      homology leakage (`split_by_identity.py` and variants)
- [ ] Decide the exact Stage-2 training mix across task families A/B and
      annotation/generation directions (not yet fixed)

## Phase 3 — Stage 2: thinking-aware SFT — **not started**

- [ ] Write the Stage-2 trainer (`src/eshmun/trainer/` was removed in the
      repo reset — this needs new training code, not a resurrection of the
      old SFT/GRPO trainers)
- [ ] Fine-tune the Stage-1 checkpoint on the thinking-augmented dataset,
      condition (a): with the `<think>...</think>` block
- [ ] Fine-tune the same checkpoint on the same ground-truth pairs, condition
      (b): without the rationale block, for a controlled comparison

## Phase 4 — Evaluation: thinking vs. direct vs. teacher — **not started**

- [ ] Head-to-head: (a) thinking-SFT student vs. (b) direct-SFT student vs.
      (c) zero-shot teacher, on identical held-out annotation and generation
      tasks
- [ ] Metrics: task accuracy/quality per task type, generation validity
      (pLDDT — pilot methodology already exercised, see `docs/experiments.md`),
      plus reasoning-trace faithfulness (does the final answer actually follow
      from the stated reasoning?)
- [ ] Ablate: does reasoning quality/coverage correlate with downstream task
      improvement?

## Phase 5 — Stretch: RL refinement (optional)

- [ ] If Phase 4 shows a benefit, explore GRPO to refine reasoning traces
      beyond what rule-based templates can express. No GRPO trainer exists in
      the current tree (removed in the repo reset, same as Phase 3's SFT
      trainer) — this would need to be written, not resumed.

---

## Out of Scope (This Phase)

- Full Eshmun model family (Eshmun-Base/Instruct/Drug, diffusion editing) —
  deferred until this hypothesis is validated
- Web interface or API serving
- Clinical or therapeutic validation

---

*Eshmun — named after the Phoenician god of healing. Kothar — named after the
Phoenician/Ugaritic god of craftsmanship.*
