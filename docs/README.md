# docs/

Working notes for the `eshmun` repo, kept as the primary source (alongside
`git log`/commit messages) for writing the manuscript at
`~/work/phd/writing/thinking_plm` ("Thinking Before Answering: Rationale-Augmented
Fine-Tuning for a Compressed Protein Language Model"). Each file below maps roughly
to one section of that paper.

- [`architecture.md`](architecture.md) — the Eshmun model itself (methods: model)
- [`data-preparation.md`](data-preparation.md) — KG construction, identity splitting,
  reasoning-trace/instruction-pair generation (methods: data)
- [`kothar-pipeline.md`](kothar-pipeline.md) — the Kothar training plan: student
  construction, Stage 1 (continued pretraining), planned Stage 2 (thinking vs.
  direct SFT), and the decision history behind the current design (methods:
  training)
- [`experiments.md`](experiments.md) — concrete results logged so far: teacher
  diagnostics, Stage-1 validation-perplexity trend, generation/pLDDT pilot (results)
- [`status.md`](status.md) — living snapshot of what's currently running/where
  things stand; check this first, it's the piece most likely to be stale

Conventions: dates are absolute (`2026-08-21`, not "last week"); numbers are
copied from their source (wandb run, CSV, script docstring) rather than
paraphrased, so a reader can go verify them; every decision that reversed an
earlier one says what the earlier one was and why it changed, since that's
exactly the kind of thing a paper's methods section needs to justify.

See also `../CLAUDE.md` (repo orientation for coding work) and `../ROADMAP.md`
(phase-by-phase plan).
