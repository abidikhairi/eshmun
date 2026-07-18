# Eshmun Roadmap

> Thinking-aware instruction tuning for protein generation and annotation.

**Horizon:** Short-term · 2–4 months
**Status:** Early Development
**Last updated:** 2026-07-17

---

## Hypothesis

InstructProtein and ProLLaMA — two state-of-the-art protein instruction models — both train
directly on `(instruction, response)` pairs. We hypothesize that inserting an explicit
chain-of-thought step before the final answer, grounded in the sequence's biological
features, improves both:

- **Generation** — text/family description → protein sequence
- **Annotation** — protein sequence → functional/family description

over direct instruction tuning, without requiring an external teacher model or RL-based
self-sampling to produce the reasoning traces.

## Approach

Reasoning traces are constructed **programmatically** from known sequence features (motifs,
domains, superfamily/family membership, composition statistics, length, etc.) rather than
distilled from a stronger LLM or self-taught (STaR). This keeps dataset construction fully
controllable and reproducible, at the cost of being limited to reasoning patterns that are
programmatically expressible.

---

## Phase 1 — Baseline replication (direct SFT)

- [ ] Curate an instruction dataset covering the same task families as InstructProtein/ProLLaMA: sequence → annotation (family, function, GO-style description) and text → sequence (family/superfamily-conditioned generation)
- [ ] Fine-tune Eshmun (decoder) directly on `(instruction, response)` pairs with the existing SFT trainer (`trainer/sft/`) — this is the baseline to beat
- [ ] Evaluate: annotation quality (text metrics), generation validity (designability proxies), perplexity

## Phase 2 — Thinking-aware dataset construction

- [ ] Define the reasoning-trace schema per task (e.g. annotation: cite detected motifs/domains → infer family → infer function; generation: restate target family/constraints → recall characteristic motifs/composition → assemble sequence)
- [ ] Implement rule-based feature extraction (motif/domain detection, composition stats, family lookup) to drive trace construction
- [ ] Implement a templating layer that turns extracted features into a natural-language `<think>...</think>` block, followed by the final answer
- [ ] Build the thinking-augmented dataset: `(instruction, chain_of_thought, response)` triples for the same tasks as Phase 1

## Phase 3 — Thinking-aware SFT

- [ ] Fine-tune Eshmun (decoder) on the thinking-augmented dataset with the existing SFT trainer, using the `<think>...</think>` response format
- [ ] Reuse `trainer/grpo/reward_functions/format.py` primitives (`PredicateReward`, `RegexReward`) as format-compliance checks during training/eval, even though training itself stays SFT in this phase

## Phase 4 — Evaluation: thinking vs. direct

- [ ] Head-to-head comparison: Phase 1 baseline vs. Phase 3 thinking-aware model, on identical held-out annotation and generation tasks
- [ ] Metrics: task accuracy/quality per task type, plus reasoning-trace faithfulness (does the final answer actually follow from the stated reasoning?)
- [ ] Ablate: does reasoning quality/coverage correlate with downstream task improvement?

## Phase 5 — Stretch: RL refinement (optional)

- [ ] If Phase 4 shows a benefit, explore GRPO (`trainer/grpo/`) to refine reasoning traces beyond what rule-based templates can express, using the format + correctness reward functions already scaffolded in `reward_functions/`

---

## Out of Scope (This Phase)

- Full Eshmun model family (Eshmun-Base/Instruct/Drug, diffusion editing) — deferred until this hypothesis is validated
- Web interface or API serving
- Clinical or therapeutic validation

---

*Eshmun — named after the Phoenician god of healing.*
