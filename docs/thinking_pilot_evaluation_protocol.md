# Thinking-Aware Pilot — Evaluation Protocol

## 1. Purpose

Decide whether to scale the thinking-aware instruction-tuning pipeline from the ~400-entry
pilot (2200 examples: 1200 annotation + 1000 generation, `khairi/eshmun-thinking-pilot`) to
the full ~20k human SwissProt entries.

**The question this protocol answers:** does prepending a programmatic `<think>` reasoning
trace before the answer improve protein annotation and generation over directly fine-tuning
on `(instruction, answer)` pairs — the InstructProtein/ProLLaMA paradigm this project is
positioned against (see `CLAUDE.md`)? Not: "is the resulting model good in an absolute
sense," "does it beat published InstructProtein/ProLLaMA numbers," or "is it publication
ready." Those are later-phase questions; a ~400-entry pilot with a 1.3B LoRA adapter isn't
statistically positioned to answer them, and chasing them here would waste the pilot's actual
purpose, which is a fast, cheap go/no-go signal before spending ~$0.24–$2.16 in DeepSeek
calls (see prior cost estimate, `PLAN.md` step 10) and ~27x the compute on the full run.

**Out of scope for this pilot:** external benchmark comparison, structure-based validation
(AlphaFold/ESMFold plausibility), wet-lab validation, statistical significance at
publication standards. Revisit at full scale.

## 2. Conditions

Three conditions, same base checkpoint, same LoRA recipe (r=64, alpha=128, `q_proj`/`k_proj`/
`v_proj`/`out_proj` + `trainable_token_indices` on `embed_tokens`), same train/val/test split,
same evaluation harness — the *only* thing that should differ between B1 and T1 is whether
the training target includes the `<think>` block.

| ID | Description | Status |
|---|---|---|
| B0 | Zero-shot `khairi/Eshmun-Thinking-Pilot` (untrained base) | ready — just run inference |
| B1 | Direct-SFT ablation: same instructions/answers, `<think>` block stripped from training targets | **dataset built and pushed** (`annotation_non_thinking`/`generation_non_thinking` configs); not yet trained |
| T1 | Thinking-SFT (this project's approach) | **dataset built and pushed** (`annotation_thinking`/`generation_thinking` configs); first run already produced `khairi/eshmun-thinking-pilot-lora`, but that run predates the split below (trained on the full undivided pilot, no held-out data) — treat it as a pipeline sanity check only, not a valid T1 result for this protocol; retrain against the split configs |

B0 is a sanity floor, not the interesting comparison — a fine-tuned model beating a zero-shot
model tells you fine-tuning works, not that reasoning traces specifically help. **B1 vs T1 is
the comparison that actually tests the hypothesis.** Report B0 anyway; if T1 doesn't clear B0
by a wide margin, something is broken regardless of the B1 comparison.

## 3. Prerequisites

### 3.1 Entry-level train/val/test split — done (2026-07-18), with a caveat

Split by **protein entry**, not by example — an entry contributes up to 3 annotation rows
(one per target relation) and up to 3 generation rows, and splitting at the example level
would leak a protein's other facts across train/test through the `Reasoning` context, letting
the model "cheat" via memorized co-occurrence rather than genuine generalization.

Built with **different methods per direction**, not one shared split: annotation's sequence is
the model's *input*, so a plain random split risks train/test sharing near-identical
sequences (a trivial homology shortcut); generation's input is a stated property, not a
sequence, so that specific risk doesn't apply there and a random split is adequate.

- **Annotation** (`scripts/python/thinking/split_annotation_by_identity.py`, external repo) —
  MMseqs2 identity-based split, adapted from
  `/run/media/khairi/seagate/data/swissprot/split_dataset.py`. Validation entries <70%
  identity to any train entry, test entries <30% (remote homology — the harder
  generalization test). Result: train 321 (80.2%), **validation 9 (2.2%)**, test 70 (17.5%).
  **The validation set is too small to trust** (see `PLAN.md` open items) — most human
  protein pairs are either close paralogs (pulled into train by the 70% cutoff) or already
  under 30% identity by default, leaving very few in the 30–70% band. Treat validation-set
  numbers from this split as anecdotal; lean on the 70-example test set for the actual
  go/no-go decision in §7.
- **Generation** (`scripts/python/thinking/split_generation_random.py`, external repo) —
  random entry-level 80/10/10. Result: train 318 (80.1%), validation 40 (10.1%), test 39
  (9.8%) — a normal-sized validation set, no caveat here.
- Assembled into 4 HuggingFace dataset configs (`scripts/python/thinking/
  build_ablation_datasets.py`, external repo), each with train/validation/test:
  `annotation_thinking`, `annotation_non_thinking`, `generation_thinking`,
  `generation_non_thinking`. Pushed to `khairi/eshmun-thinking-pilot` as additional configs
  — the original flat/unsplit config (all 2200 rows, single `train` split) is untouched, so
  the earlier in-flight T1 run that used it isn't retroactively broken, just not valid input
  for this protocol's comparisons (see §2's T1 row).

### 3.2 Direct-SFT (non-thinking) ablation dataset — done

`build_ablation_datasets.py` builds both variants from the same source rows: `_thinking`
configs keep the `Reasoning` column, `_non_thinking` configs drop it entirely (not just
empty it) — `Entry`/`Instruction`/`Answer` are otherwise byte-identical between the two, so
the only training-target difference between B1 and T1 is the presence of the `<think>` block.

### 3.3 Training B1 and (re-)training T1 — ready, not yet run

`scripts/sft/train_thinking_pilot.py` and `notebooks/thinking_sft_lora.ipynb` (main repo)
updated with `--dataset-configs` (`annotation_thinking,generation_thinking` for T1 vs.
`annotation_non_thinking,generation_non_thinking` for B1) and `--eval-split validation` for
periodic held-out loss during training. `build_example()` handles both dataset shapes
automatically (rows without a `Reasoning` column collapse to an answer-only completion).
Same LoRA config, same hyperparameters, same number of epochs for both runs — anything else
confounds the comparison. Neither B1 nor a split-aware T1 has been trained yet.

## 4. Metrics — Annotation direction (sequence → property)

Input: protein sequence. Output: `<think>...</think>\n{answer}` (T1) or `{answer}` (B1).

### 4.1 Primary: target-value accuracy

`Answer` values are drawn from `RESPONSE_TEMPLATES` in `scripts/python/thinking/reasoning.py`
(external repo) — deterministic templates per relation, single/multi-value variants joined via
`_join_values`. Parsing the model's generated answer is therefore template matching, not open-
ended NLI: write `parse_response_text()` as the literal inverse of `build_response_text()` —
match the known prefix/suffix per relation, split the joined span back into individual values
(`", "` / `" and "` boundaries), normalize (lowercase, strip trailing punctuation).

For each held-out (protein, target_relation) test example:
- **Parseable** — did the answer match a known template at all? Track this as its own rate;
  an unparseable answer is a *format* failure, distinct from a *content* failure, and
  conflating them hides which one is actually wrong.
- **Exact match** (normalized) against the true target value set from the KG.
- **Set-based precision/recall/F1** for multi-value relations (`has_function`, `involved_in`
  can have multiple true values; partial credit matters more than binary exact-match there).

Report per-relation (`member_of` / `has_function` / `involved_in` separately, plus pooled) —
these likely have different difficulty (family names are a smaller, more templated vocabulary
than free-text GO process descriptions).

### 4.2 Reasoning format compliance (T1 only)

Reuse `RegexReward`/`PredicateReward` from
`src/eshmun/trainer/grpo/reward_functions/format.py` — they already satisfy exactly this
need. Check: does the completion start with a well-formed `<think>\n...\n</think>\n` block
before the answer (`r"^<think>\n.*?\n</think>\n"`, `re.DOTALL`)? Report the compliance rate.
If this is low, the model isn't reliably learning the format at all, and content-level
comparisons against B1 become less meaningful (you'd be comparing "T1 when it works" against
"B1 always" — report both the overall rate and the conditional accuracy given compliance).

### 4.3 Reasoning-trace hallucination rate (T1 only, diagnostic — not a pass/fail gate)

Held-out proteins are ones the model has never seen KG facts for during training — it cannot
know their true `has_domain`/`located_in`/etc. values except via whatever it can infer from
the sequence or recall from pretraining. So the interesting question isn't "did it reproduce
the exact context triples" (it structurally can't, unless it memorized SwissProt during
InstructProtein's own pretraining) — it's **does it fabricate specific, checkable, wrong
facts, or does it hedge/stay vague?**

Parse `(protein, relation, value)`-shaped lines from the generated `<think>` block (same
format as `format_triple()`'s output). For each parsed triple where we have ground truth
(all relations are in the KG, held-out or not — we just don't train on the held-out ones):
- **True**: matches a real KG fact for that protein.
- **False/hallucinated**: contradicts the KG (states a relation/value combination that's
  false for that protein).
- **Unverifiable**: not parseable as a clean triple, or free-text not matching any KG value.

A high false-triple rate is a real problem for a "thinking-aware" model whose entire premise
is that the reasoning should be *grounded* — track this even though it's secondary to
answer accuracy, and mention it explicitly in the go/no-go writeup rather than only reporting
the aggregate accuracy number.

## 5. Metrics — Generation direction (property → sequence)

Input: a stated property (family/function/process). Output: `<think>...</think>\n<protein>...
</protein>` (T1) or `<protein>...</protein>` (B1).

Reuse `src/eshmun/trainer/grpo/reward_functions/protein.py` directly — it already parses
`<protein>...</protein>` + strips the `Ƥ` prefix, matching this dataset's exact encoding.

| Metric | Reuse | Notes |
|---|---|---|
| Validity (alphabet + tags) | `SequenceValidityReward` | hard filter — invalid sequences shouldn't count toward similarity below |
| Length plausibility | `LengthReward` | use the held-out protein's true length ±generous margin, or a broad default (20–2000) if not doing per-example bounds |
| Degenerate-repeat check | `AminoAcidRepetitionPenalty` | catches collapse to e.g. `AAAAA...` |
| Similarity to reference | `SequenceSimilarityReward` | reference = the held-out protein's own true sequence (see caveat below) |

**Caveat on similarity-to-reference:** the pilot reuses each protein's own real sequence as
the generation target for its own property (a documented scope simplification in `PLAN.md`,
not full per-family exemplar aggregation). So "similarity to reference" here is stricter than
it should conceptually be — many sequences could validly belong to "the PRAME family" without
resembling this *one* specific held-out example. Treat this metric as a lower bound / rough
signal, not a hard target. It's still useful comparatively (T1 vs B1 on the *same* references),
just don't over-interpret the absolute numbers.

**Novelty/memorization check:** additionally compute each generated sequence's identity
against its *nearest neighbor in the training set* (not just the matching held-out reference).
If a model scores high similarity to the reference by copying a different memorized training
sequence rather than generating something new and appropriate, that's overfitting dressed up
as a good similarity score — worth distinguishing from genuine generalization.

**Format compliance:** same `<think>` well-formedness check as §4.2, T1 only.

## 6. Statistical approach given small N

With a 40-entry test split (×3 relations ≈ up to 120 annotation examples, similarly for
generation), this is not a regime for strong significance claims. Treat it as a decision
tool, not a paper result:

- **Paired comparison, not independent-groups.** T1 and B1 are evaluated on the *identical*
  held-out examples — use a paired bootstrap (resample example indices, not per-condition)
  or a sign test on per-example win/loss, which has more power than an unpaired test at this N.
- Report point estimates with bootstrap 95% CIs on all headline numbers (accuracy, F1,
  validity rate). If the T1-vs-B1 CI on the primary annotation-accuracy metric includes 0,
  say so plainly rather than eyeballing the point estimate as a win.
- **Sanity-check before trusting any of this:** confirm the training run actually converged
  reasonably (loss curve trending down without NaN/inf/plateau-at-initialization — the current
  T1 run's loss 5.64 → 3.63 → 2.45 over 30 steps looks like normal early LoRA convergence, but
  check the full curve before evaluating, not just the first 30 steps) and that
  `save_embedding_layers` behavior (§ prior discussion) actually persisted the trained
  `<think>`/`</think>` embeddings in the saved checkpoint being evaluated.

## 7. Decision rubric

A starting point — adjust thresholds once you see real numbers, these aren't derived from any
principled statistical power calculation given the N:

- **Scale up** if: T1's paired annotation accuracy is not below B1's (within the bootstrap CI,
  i.e. no significant regression) AND `<think>` format compliance is reasonably high (say
  >80%) AND generation validity isn't majority-broken (say >50% valid sequences) for both
  conditions.
- **Iterate, don't scale yet** if: T1 clearly beats B0 (fine-tuning works) but is statistically
  indistinguishable from or worse than B1 (thinking isn't helping, or the format-compliance
  rate is too low to tell) — this points at dataset/prompt/hyperparameter issues worth fixing
  before spending the 20k-scale budget, not at abandoning the hypothesis.
- **Don't scale, rethink** if: T1 underperforms B1 with a CI that excludes 0, or hallucination
  rate is high and uncorrelated with anything useful, or generation is majority-invalid
  regardless of condition (points at a training/data pipeline bug, not a hypothesis result).

## 8. Cost/compute tie-in

Scaling instruction humanization to ~20k entries (×3 relations ≈ 60k DeepSeek calls) was
estimated at ~$0.24–$2.16 (`PLAN.md` step 10) — cheap regardless of outcome. SFT compute scales
roughly with example count: 20k×3 / 2200 ≈ 27x the current pilot's training compute. On the
6GB local GPU this pilot was originally scoped against, full-scale float32 LoRA training is
not feasible without 4-bit quantization or a bigger instance (hence training on Colab). Factor
Colab compute-unit cost into the decision alongside the quality signal — a pilot that's only
marginally positive may not justify 27x the compute spend even if it technically clears the
rubric above.

## 9. Implementation checklist

1. ~~Entry-level train/val/test split (§3.1)~~ — **done**, see §3.1 (annotation validation
   set is undersized, noted there and in `PLAN.md`).
2. ~~Non-thinking ablation dataset (§3.2)~~ — **done**, see §3.2.
3. `parse_response_text()` (§4.1) — inverse of `build_response_text()`, external repo
   (`thinking/reasoning.py` is the natural home, alongside its tests). Not built yet.
4. Reasoning-triple parser + KG cross-check (§4.3) — external repo, needs the KG. Not built yet.
5. Train B1, and retrain T1 against the split configs (the current `khairi/eshmun-thinking-
   pilot-lora` predates the split — see §2) — `scripts/sft/train_thinking_pilot.py`
   `--dataset-configs` is ready for both; runs not started.
6. Inference + scoring script — main repo (`scripts/sft/`), loads each condition's checkpoint,
   generates on the test split, applies §4/§5 metrics (reusing `reward_functions/protein.py`
   directly; format/hallucination checks are new but small). Not built yet.
7. Report: point estimates + bootstrap CIs, per-relation breakdown, decision against §7.
