# Data preparation (`scripts/data/thinking/`)

Standalone, reusable building blocks for turning UniProt/SCOP source data into
`(instruction, reasoning, answer)` triples for rationale-augmented instruction
tuning (Kothar Stage 2 — see [`kothar-pipeline.md`](kothar-pipeline.md); not
wired to any trainer yet). Everything here is adapted from an earlier
InstructProtein-pilot module of the same design (external repo,
`scripts/python/thinking/reasoning.py`), which already validated the approach:
its thinking-tuned checkpoint beat its zero-shot baseline on generation-
direction extraction rate and pLDDT. `reasoning.py`'s docstring is the primary
citation for that prior result.

## Reasoning-trace construction (`reasoning.py`)

Pure logic, no I/O — deliberately kept unit-testable. Central data type is a
`Triple(relation, value, subject="protein")`, formatted as
`(subject, relation, value)`.

- **`build_reasoning_block(triples, rng)`** — shuffles the given triples (order
  carries no meaning: they're independent observations, not a deduction chain,
  so the model shouldn't learn a fixed positional pattern) and wraps them as
  `<think>\n(...)\n(...)\n</think>`.
- **`select_context_triples(triples_by_relation, target_relation, rng, max_per_relation=3)`**
  — every relation *except* the target one, capped at 3 sampled values per
  relation so one chatty relation (e.g. `involved_in` with 15 GO terms) doesn't
  drown out the others.
- **`build_response_text(target_relation, values)`** — the final answer,
  filled from a small per-relation template dict (`RESPONSE_TEMPLATES`), with
  separate singular/plural phrasing. Covers `member_of`, `has_function`,
  `involved_in`, `located_in`, `catalyzes`, and the three SCOP relations
  (`scop_fold`, `scop_superfamily`, `scop_family` — the SCOP ones get a
  "(SCOP classification)" suffix since e.g. `scop_family` and SwissProt's
  `member_of` family string are different concepts that happen to share the
  English word "family").

### Residue encoding

**`encode_sequence(sequence)`** wraps a raw AA sequence as
`<protein>ƤAƤCƤD...</protein>` — every residue individually prefixed with `Ƥ`,
because `hicai-zju/InstructProtein`'s tokenizer has a dedicated single-token
vocabulary entry per `Ƥ`+residue (`ƤA`, `ƤC`, ...) and this is the only way to
land on those tokens instead of falling back to ambiguous plain-English BPE.

A real bug here was found and fixed (commit `7893758`): the naive
`Ƥ.join(sequence)` only inserts `Ƥ` *between* residues, so the first residue of
every sequence tokenized as plain English `'A'` (token id 250) instead of the
dedicated `ƤA` — confirmed empirically by encoding all 20 standard AAs and
inspecting token ids. Fixed to prefix every residue including the first. This
affects every consumer of `encode_sequence` (`build_task_a_annotation.py`,
`build_task_a_generation.py`, `build_ppi_dataset.py`, `build_scop_dataset.py`,
and `scripts/kothar/build_pretrain_mix.py`'s UniRef50 encoding) — none had
generated output yet when the fix landed, so nothing needed regenerating.

Revisit if training ever moves to Eshmun's own tokenizer rather than
InstructProtein's — a from-scratch tokenizer would need its own protein-token
vocabulary story (see [`architecture.md`](architecture.md#relationship-to-the-tokenizer-ablation-paper)
for why AA-level tokenization alone is not expected to be a quality problem).

## Two task families

**Task family A — human annotation/generation** (`build_task_a_annotation.py`,
`build_task_a_generation.py`), built from a UniProt-derived knowledge graph
over Homo sapiens entries (`build_annotation_kg.py`; also
`build_all_organism_*` variants exist for a broader-organism KG).

- **Annotation direction** (sequence → text): input is a real sequence, so
  reasoning is allowed to cite that entry's *other* real facts (the remaining
  target relations, plus `has_domain`/`has_region`/`has_motif`/`has_length`
  as context) — legitimate, since nothing about the input is hidden.
  `interacts_with` is deliberately excluded from context (same rationale as
  the PPI builder below): a specific interaction partner isn't a
  generalizable single-protein trait, and citing it would invite a shortcut
  ("already interacts with lots of things") instead of content-based
  reasoning.
- **Generation direction** (text → sequence): the target relation
  (family/function/process/localization/catalytic activity) is the *input*,
  not a hidden answer, so reasoning here cites **category-level aggregate
  statistics** (`build_generation_kg.py`, computed from train-split entries
  only — the causal-inversion / leakage fix), not instance-specific facts
  about the one sequence being generated. The generation target is the
  category's real member sequences (documented scope simplification — not a
  per-category aggregated exemplar; revisit at full scale). Instructions are
  sampled from a DeepSeek-generated pool
  (`generate_generation_instruction_pool.py`,
  `generation_instruction_pool.json`) rather than the ~3-variant hand-written
  templates in `reasoning.py`, for phrasing diversity.

**Task family B — SCOP/PPI** (`build_scop_dataset.py`, `build_ppi_dataset.py`,
`build_scop_kg.py`), same reasoning-trace machinery applied to structural
classification (SCOP fold/superfamily/family) and protein-protein interaction
data, across all organisms (not human-only).

Both families' example construction is seeded deterministically per-example
(`rng_for(*parts)` = a SHA-256 hash of the example's identifying strings, e.g.
`("task_a_annotation", entry, target_relation)`, reduced mod 2**32) —
reproducible without a shared global RNG state across a parallelizable build.

## Identity-based train/val/test split (`split_by_identity.py`, `split_scop_dataset_by_identity.py`, `split_all_organism_by_identity.py`)

Splits are by **sequence identity via MMseqs2**, not random — annotation's
sequence is the model's *input*, so a random split risks train/test sharing
near-identical sequences and letting the model cheat via sequence-similarity
shortcut instead of genuine generalization. Methodology (same thresholds as
the InstructProtein pilot, here run at full scale rather than a 400-entry
pilot sample):

1. Random candidate 80/20 train/holdout split.
2. Any holdout entry ≥70% identical to a train entry (`mmseqs easy-search`,
   `--min-seq-id 0.70 -c 0.8 --cov-mode 0`) is moved back into train — this
   defines a "near" validation set from what's left (<70% identity to train).
3. Any remaining validation candidate ≥30% identical to a train entry is
   pulled into the **test** set instead — a "far", remote-homology
   generalization test (<30% identity to train). Candidates that clear the
   70% check but not the 30% check aren't a fair remote-homology test case, so
   they stay in validation rather than being dropped.
4. Length cap: sequences must be <512 residues (locked scope decision — the
   dataset covers proteins under 512 residues only).

Output: `data/thinking/processed/annotation_split.csv` (or the SCOP/
all-organism equivalents), columns `(entry, split)`.

One documented pitfall to avoid regenerating: `homosapiens-sequences.tsv` (not
`swissprot_sequence_features.parquet`, a leftover from an unrelated, older
MLM-era pipeline that's silently pre-filtered to ≤400 residues) is the correct,
unfiltered sequence source.

## Instruction-pool humanization

`generate_instruction_pool.py` / `generate_generation_instruction_pool.py`
build `instruction_pool.json` / `generation_instruction_pool.json` — larger,
DeepSeek-generated phrasing pools per relation, used in place of `reasoning.py`'s
own hand-written ~3-variant template dicts wherever more diversity is wanted
(annotation direction always uses the pool; generation direction now uses it
too — see the docstring note in `build_task_a_generation.py` for why generation
originally *didn't* get this treatment and why that turned out to be a weaker
argument than it first looked).

## KG-to-graph-DB loading

`load_neo4j_task_a_annotation.py` / `load_neo4j_remaining.py` load the
constructed KGs into Neo4j — used for exploration/validation of the KG itself
(e.g. checking relation coverage, category sizes), not part of the training
data path.

## Not yet wired to a trainer

All of the above produces `.jsonl`/`.parquet` files
(`data/thinking/processed/task_a_{annotation,generation}_pairs.jsonl`, plus
the SCOP/PPI/all-organism equivalents) — reusable building blocks for whatever
Stage 2 (thinking-aware SFT) trainer gets written next; `src/eshmun/trainer/`
was removed in the repo reset (see [`status.md`](status.md)) and hasn't been
rebuilt.
