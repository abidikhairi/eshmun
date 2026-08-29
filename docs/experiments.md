# Experiments and results log

Concrete numbers produced so far, in roughly chronological/pipeline order.
Each entry names the script that produced it, so it can be reproduced or
extended rather than taken on faith.

## Teacher natural-language diagnostic (2026-08-21/22)

**Motivation**: before committing to "InstructProtein as eval-only baseline,
no teacher-generated training data" (see
[`kothar-pipeline.md`](kothar-pipeline.md#decision-history)), checked how well
the teacher actually handles the natural-language side of the Stage-1 replay
mix.

`scripts/kothar/eval_pretrain_mix_perplexity.py` — 10 samples/subset from
`data/kothar/pretrain_mix.parquet`, 150-token truncation (the 1.3B teacher in
float32 barely fits this machine's 6GB GPU; OOM'd at the model's full
2048-token limit):

| subset | mean perplexity |
|---|---:|
| UniRef50 | 15.95 (sane) |
| PubMed | in the billions |
| FineWeb-Edu | in the billions |
| FineMath | in the billions |

The text-subset numbers are *worse than random guessing*
(`ln(vocab_size)=ln(50304)≈10.8` nats/token would be random). Verified this
is real, not a measurement bug: identical loss from manual cross-entropy, HF's
built-in loss, and both the Hub and a local copy of the checkpoint. Root cause
(via greedy decoding): the teacher treats plain English as an incomplete
generation prompt and immediately starts emitting `ƤX` residue tokens instead
of continuing in English — even for template sentences straight out of
`reasoning.py`'s own response templates, e.g.
`"The protein belongs to the"` → `...theƤLƤLƤAƤLƤGƤIƤNƤHƤKƤTƤAƤPƤVƤSƤL`.

A follow-up, `scripts/kothar/eval_protein_space_perplexity.py`, restricts the
softmax to just the 20 residue tokens at positions whose true label is a
residue (factoring out the "does it even try to speak English" confusion) and
gets **16.16/20** overall — consistent with the full-vocab UniRef50 number,
and in line with typical unconditioned protein-LM next-residue perplexity
(high intrinsic entropy, not alarming on its own).

**Framing for the paper**: this is a concrete, positive argument for Stage 1's
natural-language replay data — it exists specifically to counteract this weak
grip on free-form English, which a protein-only continued-pretrain would not
address and might worsen.

## Student construction verification (2026-08-21)

`build_student.py`'s `sanity_forward` check: the 409M student produces finite
logits on a trivial forward pass immediately after layer-subsetting + special
token resize, and reloads with matching tokenizer ids. See
[`kothar-pipeline.md`](kothar-pipeline.md#stage-0--student-construction-build_studentpy-done-2026-08-21)
for construction details. No quantitative eval at this stage (that's the point
of Stage 1) — this only confirms the construction didn't silently break the
model.

## Stage-1 validation-perplexity trend (checkpoints 1000–3750)

Steps 1000–2500 sourced from wandb run `flursky/Kothar/xvxcbijd`
("ppl-by-source-checkpoints-1000-1500-2000", extended to include 2500); steps
2750 onward added 2026-08-28/29 by re-running `checkpoint_trend.py` directly
(not logged to wandb this time — that run's wandb-table logging wasn't part
of the committed script, just an ad hoc addition in the session that
produced it). All runs are against `data/kothar/valid_holdout.parquet`
(disjoint from the training mix, same 10:5:3:1 source ratio, 190 rows); local
copy of the raw per-step CSV at `data/kothar/checkpoint_trend.csv`. The
2000/2500 rows were re-verified bit-for-bit against the wandb-logged values
while producing the 2750/3000 points, confirming the eval is
deterministic/reproducible.

| step | uniref50 | pubmed | finemath | fineedu | overall |
|---:|---:|---:|---:|---:|---:|
| 1000 | 16.57 | 234.85 | 203.52 | 400.71 | 58.49 |
| 1500 | 16.45 | 164.83 | 129.65 | 290.48 | 48.61 |
| 2000 | 16.34 | 129.03 | 93.31 | 228.51 | 42.58 |
| 2500 | 16.21 | 107.91 | 72.31 | 189.41 | 38.48 |
| 2750 | 16.13 | 99.39 | 64.84 | 174.67 | 36.76 |
| 3000 | 16.11 | 92.94 | 59.40 | 164.01 | 35.47 |
| 3250 | 16.06 | 87.26 | 54.72 | 155.17 | 34.29 |
| 3500 | 16.03 | 81.82 | 50.20 | 145.89 | 33.11 |
| 3750 | 15.99 | 77.18 | 47.01 | 139.15 | 32.15 |

Reading this: UniRef50 perplexity is already low and barely moving (16.6→16.0,
close to the teacher's own 15.95 baseline above — makes sense, the student
warm-starts from teacher layers, and protein modeling was already the
teacher's strength). The three natural-language subsets are all improving
steadily and substantially (pubmed 235→77, finemath 204→47, fineedu 401→139)
— i.e. the replay mix is doing its job of recovering natural-language
competence, and hasn't plateaued as of step 3750. Overall held-out perplexity
dropped from 58.5 to 32.1 over the same range. Training will be stopped at
step 5250 (epoch 5) regardless of whether a plateau is reached by then — see
[`status.md`](status.md).

**Overfitting check**: cross-referencing against training loss at the same
steps (`train/loss` from wandb run `ltiumvoj`, converted to perplexity):

| step | train ppl | validation ppl (overall) | gap |
|---:|---:|---:|---:|
| 2000 | 40.37 | 42.58 | 2.21 |
| 2500 | 37.77 | 38.48 | 0.71 |
| 2750 | 35.62 | 36.76 | 1.14 |
| 3000 | 35.36 | 35.47 | 0.11 |
| 3250 | 34.44 | 34.29 | −0.15 |
| 3500 | 32.41 | 33.11 | 0.70 |
| 3750 | 31.57 | 32.15 | 0.58 |

The gap stays small (≤2.2 points) and isn't widening with training — if the
model were starting to memorize training data, train loss would drop faster
than held-out loss and this gap would grow steadily instead. No overfitting
signal yet, but this is still only ~28% of the planned 10-epoch schedule;
FineMath in particular was
sampled with replacement into the training mix (`kothar-pipeline.md`) and is
the subset most likely to show overfitting first if it happens. **Not yet a
plateau** — Stage 1 was still running as of this writing (see
[`status.md`](status.md)); this table should be regenerated with later
checkpoints before drawing a "Stage 1 is done" conclusion.

## Generation pilot: unconditioned vs. Met-prefixed (checkpoint-2500)

`scripts/kothar/generate_sequences.py`, checkpoint-2500, 30 sequences,
`do_sample=True, top_p=0.95, top_k=250, repetition_penalty=1.3,
max_new_tokens=512`, seed 4242.

- **Unconditioned** (`--prefix ""`): only 16/30 sequences closed
  `</protein>` within the token budget. Inspecting the 14 incomplete rows
  (`data/generations/checkpoint-2500_gen30.fasta`) shows this isn't just
  running out of budget mid-residue: several samples fully exit "residue
  mode" and generate **plain English prose** instead (e.g. one sample's
  post-`<protein>` continuation reads *"...in the factoring characteristics
  are described by using long-term use as a secondary mechanism of the use,
  with relative role could be investigated."*), before the model's own EOS
  fires. A concrete, checkpoint-level signal that Stage 1 — domain-adaptive
  pretraining, not generation-instruction-tuned — hasn't yet learned to
  reliably stay in protein-generation mode once inside `<protein>...</protein>`;
  worth re-checking at later checkpoints and after Stage 2 (task conditioning
  should make the boundary explicit rather than implicit).
- **`M`-prefixed** (`--prefix M`, i.e. seed with Met, the near-universal
  translation-initiation residue): 30/30 completed, no prose drift observed.
  Purely a generation-time prompting change, not a training change — included
  here because it's a concrete, reproducible knob on sample quality/validity
  worth citing if the paper discusses generation-direction evaluation
  protocol.

**Bug found and fixed while inspecting this (2026-08-28)**: those same 14
incomplete rows also had literal special-token text (`</s>`, then `<pad>`
repeated up to ~350×) appended into the "sequence" field in the saved FASTA —
`extract_sequence`'s incomplete-case path only stripped the `Ƥ` residue
marker, not other decoded special-token text (`tokenizer.decode(...,
skip_special_tokens=False)`, needed to detect an unclosed tag, also renders
`</s>`/`<pad>` as literal text; already-finished samples in a
`num_return_sequences` batch get padded to the longest sample's length).
Fixed in `generate_sequences.py` by filtering the extracted body to the valid
residue alphabet rather than only removing the `Ƥ` marker. Only affects
*incomplete* rows — every complete sequence (including both M-prefixed pilot
runs used for the pLDDT numbers above) was already unaffected, since
extraction there stops cleanly at `</protein>`. `checkpoint-2500_gen30.fasta`
was regenerated with the fixed script (same unconditioned protocol, seed
4242) to confirm: the new file's incomplete rows are clean amino-acid text,
no leaked tag text. Completion rate was 16/30 again in the regenerated run
(same as before), though the specific sampled content differs from the
original run — GPU sampling isn't bit-reproducible across separate processes
even with a fixed seed, so this is expected, not a regression.

**pLDDT on the regenerated unconditioned set (n=30, 1 failure)**: one row
(`seq_7`) came back length-0 (the model produced nothing before EOS) and was
correctly rejected by the folding API rather than silently folding garbage.
Of the 29 folded: mean=36.50, median=30.82, min=21.81 (seq_28, 398 aa),
max=89.42 (seq_11, only 4 aa — treat as an artifact, not a real signal: pLDDT
on a 4-residue fragment isn't a meaningful fold-confidence measurement, short
fragments routinely score spuriously high). Higher mean than the M-prefixed
pilot below (29.50) but not a real quality improvement — driven by several
very short incomplete fragments (4–23 aa) getting inflated scores, not by
better full-length generations. Output:
`data/eval/pretraining/checkpoint-2500_gen30_plddt_esmfold2.csv`.

## Structural plausibility pilot: pLDDT (checkpoints 2500–3750, M-prefixed, n=30 each)

Folded with ESMFold2 (EvolutionaryScale, via Biohub Forge SDK,
`scripts/eval/pretraining/plddt_esmfold2_forge.py`, model
`esmfold2-fast-2026-05`, `num_loops=3`, `num_sampling_steps=10`). The public
ESM Atlas API (`plddt_esmfold.py`) was tried first but hit persistent `504`
errors during this run (service-side outage, not a script bug) and was
abandoned in favor of ESMFold2/Forge.

All checkpoints: 30/30 sequences folded successfully, same generation
protocol (`--prefix M`, seed 4242, checkpoint's own weights at that step).

| checkpoint | mean | median | min | max |
|---|---:|---:|---:|---:|
| 2500 (`checkpoint-2500_gen30_prefixM_plddt_esmfold2.csv`) | 29.50 | 26.44 | 20.53 (seq_15, 314 aa) | 45.76 (seq_28, 77 aa) |
| 2750 (`checkpoint-2750_gen30_prefixM_plddt_esmfold2.csv`) | 30.29 | 28.80 | 20.44 (seq_15, 310 aa) | 46.02 (seq_13, 37 aa) |
| 3000 (`checkpoint-3000_gen30_prefixM_plddt_esmfold2.csv`) | 32.50 | 33.31 | 22.29 (seq_27, 242 aa) | 47.53 (seq_4, 23 aa) |
| 3250 (`checkpoint-3250_gen30_prefixM_plddt_esmfold2.csv`) | 32.46 | 31.06 | 21.50 | 47.53 |
| 3500 (`checkpoint-3500_gen30_prefixM_plddt_esmfold2.csv`) | 30.01 | 27.47 | 20.37 | 46.02 |
| 3750 (`checkpoint-3750_gen30_prefixM_plddt_esmfold2.csv`) | 30.39 | 27.44 | 20.96 | 47.51 |

Six checkpoints now: mean 29.50→30.29→32.50→32.46→30.01→**30.39**. Unlike
the perplexity trend (monotonically improving on every source through step
3750), pLDDT is **not monotonic** — it rose through 3000-3250 then has
oscillated in a ~30-32 band since (3500: 30.01, 3750: 30.39), rather than
continuing to track perplexity's steady decline. Since each point is an
independent 30-sequence sample (different random draws each time, not the
same sequences re-folded across checkpoints), this reads as sampling noise
on top of a flat-to-mildly-improving underlying trend, not a real regression
in the model — but it's a concrete demonstration that this 30-sequence pLDDT
pilot is noisier than the perplexity trend and shouldn't be read
checkpoint-to-checkpoint in isolation (a 2-3 point swing is within its noise
floor). Larger sample sizes or re-folding the same sequence set across
checkpoints would be needed to get a cleaner signal if this metric becomes
load-bearing for a paper claim.

**Read these numbers cautiously**: pLDDT in the 20–46 range is low by the
standards of *real, evolved* proteins (pLDDT is conventionally reported
0–100; well-folded natural proteins are typically 80+). This is a
pretraining-stage checkpoint sampling with no task conditioning — a proxy for
"does the model produce anything structurally plausible at all," not a claim
of designed-protein quality. Worth re-running at later Stage-1 checkpoints and
again after Stage 2 (task-conditioned generation) for a real trend, rather
than treating this one pilot number as a final result.
