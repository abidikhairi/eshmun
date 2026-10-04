"""Reusable pieces of the supervised fine-tuning stack.

Submodules are imported directly (`from eshmun.training.trainer import
SpanWeightedSFTTrainer`). This file deliberately imports nothing, so a bare
`import eshmun.training` does not pull in `trl` or `datasets`.
"""
