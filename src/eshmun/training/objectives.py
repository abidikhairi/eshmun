"""The span-weighted cross-entropy objective and the weights it mixes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# --- the span-weighted cross-entropy objective -------------------------------
#
# The objective averages cross-entropy inside two spans and mixes the two
# means, instead of taking a single mean over every supervised token:
#
#     L = 0.25 * mean(CE_think) + 0.75 * mean(CE_seq)
#
#   * thinking span -- the first supervised token through `</think>`
#   * sequence span -- everything after it: `<protein>`, the residues,
#     `</protein>`, and the closing `</s>`, so the end-of-turn token keeps
#     being trained
#
# The weights sit where the corpus' own loss mass already is: on the small
# split, 25.7% of labeled tokens fall in the thinking span and 74.3% in the
# sequence span (measured over 500 rows, closers included, which is exactly how
# this partition cuts them). So the mixture stays put and the split just
# becomes explicit -- and turnable -- instead of implicit in the token counts.
#
# Inside a span the tokens are not weighted equally: the token that *closes*
# the span -- `</think>` in the thinking span, `</protein>` in the sequence
# span -- carries STRUCTURAL_TOKEN_WEIGHT instead of 1. An early stop is one
# wrong token in a row of ~150, worth 0.7% of the sequence span under a plain
# mean and ~3% at 5. Each span is a weighted mean of its own tokens, so an
# upweighted token moves mass inside its span rather than rescaling the whole
# loss.
THINK_LOSS_WEIGHT = 0.25
SEQUENCE_LOSS_WEIGHT = 0.75
STRUCTURAL_TOKEN_WEIGHT = 5.0


def span_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    think_close_id: int,
    protein_close_id: int,
    think_weight: float = THINK_LOSS_WEIGHT,
    sequence_weight: float = SEQUENCE_LOSS_WEIGHT,
    structural_weight: float = STRUCTURAL_TOKEN_WEIGHT,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The objective above, plus each span's plain mean CE for logging.

    `logits` are the model's full, unsliced outputs. That is what the hook
    receives: with `compute_loss_func` set the trainer forwards without
    `labels`, so nothing slices them down to the supervised positions, and the
    two tensors line up as `logits[:, :-1]` / `labels[:, 1:]`.

    The returned loss is a mean over the batch; a caller under gradient
    accumulation scales it by its share of `num_items_in_batch` (see
    `SpanWeightedSFTTrainer`). The spans are pooled over the batch, not per
    row, and a batch with an empty span hands the other span the full weight,
    so neither case can divide by zero.
    """
    # `logits[:, t]` predicts `labels[:, t + 1]`. Spans are decided on the
    # predicted token, so a closer belongs to the span it closes.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    supervised = shift_labels != ignore_index

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1).long(),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(shift_labels.shape)

    # The *first* `</think>` is the boundary; a repeat later in the row belongs
    # to the sequence span it sits in, not back to the thinking span. With no
    # `</think>` at all the whole row falls in the thinking span, which the
    # weights then renormalize into its plain mean.
    at_think_close = shift_labels == think_close_id
    at_protein_close = shift_labels == protein_close_id
    think_seen = at_think_close.cumsum(dim=1)
    protein_seen = at_protein_close.cumsum(dim=1)
    first_think_close = at_think_close & (think_seen == 1)
    first_protein_close = at_protein_close & (protein_seen == 1)
    in_think = (think_seen == 0) | first_think_close
    in_sequence = ~in_think

    weight = torch.ones_like(per_token)
    weight[first_think_close | first_protein_close] = structural_weight
    weight = weight * supervised

    think_mass = (weight * in_think).sum()
    sequence_mass = (weight * in_sequence).sum()
    think_terms = per_token * weight * in_think
    sequence_terms = per_token * weight * in_sequence
    think_mean = think_terms.sum() / think_mass.clamp(min=1)
    sequence_mean = sequence_terms.sum() / sequence_mass.clamp(min=1)

    think_on = think_weight if bool(think_mass > 0) else 0.0
    sequence_on = sequence_weight if bool(sequence_mass > 0) else 0.0
    total = think_on + sequence_on
    if total == 0.0:
        # Nothing supervised in this batch at all: a zero that still carries
        # the graph, so the step is a no-op rather than a crash.
        loss = per_token.sum() * 0.0
    else:
        loss = (think_on * think_mean + sequence_on * sequence_mean) / total

    with torch.no_grad():
        think_plain = (per_token * supervised * in_think).sum()
        sequence_plain = (per_token * supervised * in_sequence).sum()
        think_count = (supervised & in_think).sum().clamp(min=1)
        sequence_count = (supervised & in_sequence).sum().clamp(min=1)
        diagnostics = {
            "ce_think": (think_plain / think_count).item(),
            "ce_seq": (sequence_plain / sequence_count).item(),
        }
    return loss, diagnostics
