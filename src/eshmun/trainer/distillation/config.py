"""Configuration for knowledge distillation training."""

from dataclasses import dataclass

from eshmun.trainer.sft.config import SFTConfig


@dataclass
class DistillationConfig(SFTConfig):
    """
    Training configuration for DistillationTrainer. Extends SFTConfig with
    knowledge-distillation-specific fields.

    Args:
        temperature: Softmax temperature applied to both student and teacher logits
            when computing the soft-target KL loss. Higher values produce softer
            distributions. Hinton et al. recommend 2–8 for typical tasks.
        alpha: Interpolation coefficient between the hard-label CE loss and the
            soft-target KD loss: ``total = alpha * kd_loss + (1 - alpha) * ce_loss``.
            Set to 1.0 to train purely on teacher soft targets; 0.0 for pure CE.
        distill_hidden_states: If True, add an MSE penalty between corresponding
            student and teacher hidden states.  Requires the student and teacher
            to expose ``hidden_states`` in their outputs (pass
            ``output_hidden_states=True``).
        distill_attentions: If True, add an MSE penalty between corresponding
            student and teacher attention maps.  Requires ``output_attentions=True``
            in model outputs.
        teacher_layer_map: Explicit list mapping student layer indices to teacher
            layer indices for the hidden-state and attention losses.  If None,
            layers are matched by position (student layer i → teacher layer i).
            Must be provided when the student and teacher have different depths.
        hidden_loss_weight: Scalar weight applied to the hidden-state MSE loss term.
        attention_loss_weight: Scalar weight applied to the attention MSE loss term.
    """

    temperature: float = 4.0
    alpha: float = 0.5
    distill_hidden_states: bool = False
    distill_attentions: bool = False
    teacher_layer_map: list[int] | None = None
    hidden_loss_weight: float = 1.0
    attention_loss_weight: float = 1.0
