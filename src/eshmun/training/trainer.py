"""`SFTTrainer` whose objective is `span_weighted_cross_entropy`."""

from __future__ import annotations

from typing import Any

import torch
from trl import SFTTrainer

from eshmun.training.objectives import span_weighted_cross_entropy


class SpanWeightedSFTTrainer(SFTTrainer):
    """`SFTTrainer` whose objective is `span_weighted_cross_entropy`.

    The objective is installed through `compute_loss_func` rather than by
    overriding `compute_loss`. With that hook set the trainer sets the labels
    aside before the forward, so the plain cross-entropy is replaced rather
    than added to -- it is never computed -- while TRL's entropy and
    token-accuracy metrics still come out of the same forward.

    The hook also changes the gradient-accumulation convention: with it set,
    `Trainer.training_step` stops dividing a micro-batch's loss by the
    accumulation steps and expects the function to use `num_items_in_batch`
    itself. The loss returned here is scaled by the micro-batch's share of the
    accumulation window's supervised tokens -- the scaling the model's own
    summed loss carries -- which also keeps the logged `loss` on the scale of
    a plain mean.
    """

    def __init__(
        self,
        *args: Any,
        think_close_id: int,
        protein_close_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.think_close_id = think_close_id
        self.protein_close_id = protein_close_id
        # A bound method, so `Trainer.compute_loss` can call it exactly as
        # `compute_loss_func(outputs, labels, num_items_in_batch=...)`.
        self.compute_loss_func = self._span_weighted_loss

    def _span_weighted_loss(
        self,
        outputs: Any,
        labels: torch.Tensor,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        loss, diagnostics = span_weighted_cross_entropy(
            outputs.logits, labels, self.think_close_id, self.protein_close_id
        )
        mode = "train" if self.model.training else "eval"
        for name, value in diagnostics.items():
            self._metrics[mode][name].append(value)

        if num_items_in_batch is not None:
            # Counted the way `Trainer._get_num_items_in_batch` counts it, so
            # the two shares agree.
            counted = labels[..., 1:] if self._loss_shifts_labels else labels
            loss = loss * (counted != -100).sum() / num_items_in_batch
        return loss
