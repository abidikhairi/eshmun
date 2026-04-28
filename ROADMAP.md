# Eshmun Roadmap
> A series of protein language models spanning understanding, generation, reasoning, and drug design.

**Horizon:** Short-term · 3–6 months  
**Status:** Early Development  
**Last updated:** April 2026

---

## Vision

Eshmun is a family of protein language models built along two axes: **understanding** (encoder-style, biologically grounded) and **generation & reasoning** (decoder-style, instruction-following, chain-of-thought). The series progressively scales from pure protein understanding to multi-modal reasoning over proteins, molecules, and natural language — culminating in specialized models for drug design and protein editing.

---

## Model Family Overview

| Model | Paradigm | Description |
|---|---|---|
| **Eshmun-Zero** | Encoder (MLM) | Protein understanding LM with local + global attention |
| **Eshmun-GPT** | Decoder (CLM) | Generative protein model trained à la GPT-2 |
| **Eshmun-Base** | Encoder-Decoder | Distilled from Qwen; joint language & protein understanding |
| **Eshmun-Instruct** | Decoder (IT) | Instruction-following model for protein, molecule & text generation |
| **Eshmun-Drug** | Decoder (IT) | Instruction-following model specialized for drug design |
| **Eshmun-Zero-R1** | Encoder + Diffusion | Eshmun-Zero extended with diffusion for protein editing (thermal stability) |
| **Eshmun-R1** | Decoder (RL) | Eshmun-Instruct further tuned with chain-of-thought + GRPO |

---

## Phase 1 — Core Understanding & Generation (Month 1–2)

### Eshmun-Zero
*Protein understanding language model with local and global attention.*

- [ ] Design hybrid attention architecture: local sliding-window + global CLS/register tokens
- [ ] Curate pretraining corpus (UniRef50/90, filtering by length and quality)
- [ ] Pretrain with masked language modeling (MLM) objective
- [ ] Evaluate on: contact prediction, remote homology, secondary structure (TAPE benchmarks)
- [ ] Release pretrained weights + tokenizer

### Eshmun-GPT
*Autoregressive protein sequence generation, trained like GPT-2.*

- [ ] Define causal transformer architecture (decoder-only, protein vocabulary)
- [ ] Train on large-scale protein sequences with next-token prediction (CLM)
- [ ] Evaluate on: sequence perplexity, unconditional generation novelty & diversity, designability proxies
- [ ] Release pretrained weights + generation scripts

---

## Phase 2 — Multimodal Base & Instruction Tuning (Month 2–4)

### Eshmun-Base
*Distilled from Qwen; understands both natural language and protein sequences.*

- [ ] Set up knowledge distillation pipeline from Qwen (token-level + hidden-state alignment)
- [ ] Build mixed-modality training data: interleaved protein sequences and text descriptions
- [ ] Train on joint protein–language corpus (UniProt text annotations, PubMed abstracts, sequence–function pairs)
- [ ] Evaluate on: protein QA, sequence-to-description alignment, language perplexity
- [ ] Release base weights as foundation for Instruct and Drug variants

### Eshmun-Instruct
*Instruction-following model for protein, molecule, and text generation & understanding.*

- [ ] Curate instruction-tuning dataset: protein design tasks, molecule captioning, Q&A, sequence annotation
- [ ] Fine-tune Eshmun-Base with supervised instruction tuning (SFT)
- [ ] Add SMILES / molecular token support for small-molecule tasks
- [ ] Evaluate on: instruction-following accuracy, molecule generation validity, protein description quality
- [ ] Release instruction-tuned weights + prompt templates

### Eshmun-Drug
*Instruction-following model specialized for drug design workflows.*

- [ ] Curate drug-design instruction dataset: target-to-hit, lead optimization, ADMET prediction, binding affinity tasks
- [ ] Fine-tune Eshmun-Instruct (or Eshmun-Base directly) on drug-design corpus
- [ ] Integrate protein–ligand interaction data (ChEMBL, BindingDB, PDBbind)
- [ ] Evaluate on: molecular docking score correlation, drug-likeness (QED, SA score), target specificity
- [ ] Release drug-design weights + task-specific prompt library

---

## Phase 3 — Reasoning & Editing (Month 4–6)

### Eshmun-Zero-R1
*Eshmun-Zero encoder extended with a diffusion head for protein editing (thermal stability).*

- [ ] Design diffusion head on top of Eshmun-Zero encoder representations
- [ ] Curate protein editing dataset: wild-type / mutant pairs with thermostability labels (ΔTm, ΔΔG)
- [ ] Train diffusion model conditioned on encoder embeddings for targeted sequence editing
- [ ] Evaluate on: Δ stability prediction (ProtaBank, FireProtDB), edit naturalness, sequence recovery
- [ ] Release editing pipeline with demo for thermal stability optimization

### Eshmun-R1
*Eshmun-Instruct further tuned with chain-of-thought reasoning and reinforcement learning (GRPO).*

- [ ] Collect or synthesize chain-of-thought reasoning traces for protein and molecule tasks
- [ ] Fine-tune Eshmun-Instruct with CoT SFT as cold start
- [ ] Define reward model / reward functions for GRPO (correctness, biological validity, format)
- [ ] Run GRPO training loop; monitor KL divergence and reward stability
- [ ] Evaluate on: reasoning quality, multi-step task completion, benchmark uplift over Eshmun-Instruct
- [ ] Release final weights + GRPO training code

---

## Open-Source Release Plan

| Release | Contents | Target |
|---|---|---|
| v0.1 | Eshmun-Zero weights + tokenizer | End of Month 2 |
| v0.2 | Eshmun-GPT weights + generation scripts | End of Month 2 |
| v0.3 | Eshmun-Base + Eshmun-Instruct weights | End of Month 3 |
| v0.4 | Eshmun-Drug weights + prompt library | End of Month 4 |
| v0.5 | Eshmun-Zero-R1 editing pipeline | End of Month 5 |
| v1.0 | Eshmun-R1 weights + GRPO training code + full documentation | End of Month 6 |

---

## Publications & Dissemination

- [ ] Technical report: Eshmun-Zero architecture and benchmarks
- [ ] Technical report: Eshmun-R1 — chain-of-thought reasoning for protein tasks via GRPO
- [ ] Preprint (bioRxiv): The Eshmun model series — a unified family of protein language models
- [ ] Submit to a relevant venue (NeurIPS ML4Molecules, ICLR, RECOMB, or similar)

---

## Milestone Summary

| Milestone | Target |
|---|---|
| Eshmun-Zero pretraining complete | End of Month 2 |
| Eshmun-GPT pretraining complete | End of Month 2 |
| Eshmun-Base distillation complete | End of Month 3 |
| Eshmun-Instruct & Eshmun-Drug SFT complete | End of Month 4 |
| Eshmun-Zero-R1 diffusion training complete | End of Month 5 |
| Eshmun-R1 GRPO training complete | End of Month 6 |
| All models publicly released (v1.0) | End of Month 6 |
| Preprint submitted to bioRxiv | End of Month 6 |

---

## Out of Scope (This Phase)

- Web interface or API serving
- Clinical or therapeutic validation

---

*Eshmun — named after the Phoenician god of healing.*