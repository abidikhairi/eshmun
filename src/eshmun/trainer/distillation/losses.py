"""Loss functions for knowledge distillation."""

import torch
import torch.nn.functional as F


def soft_target_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    ignore_index: int = -100,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KL divergence between temperature-scaled student and teacher distributions.

    Implements the Hinton et al. (2015) distillation loss:
        L_KD = T² · KL(softmax(z_t / T) || log_softmax(z_s / T))

    The T² factor compensates for the gradient magnitude shrinkage caused by
    the temperature scaling.

    When ``labels`` is provided, positions where ``labels == ignore_index`` are
    excluded from the loss (same masking convention as CrossEntropyLoss).

    Args:
        student_logits: (B, T, V) or (B, V) student logits.
        teacher_logits: Same shape as student_logits.
        temperature: Softmax temperature T > 0.
        ignore_index: Token index to mask out (typically the padding id).
        labels: (B, T) or (B,) label tensor for masking.  If None, all positions
            are included.

    Returns:
        Scalar KL loss.
    """
    T = temperature

    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)

    kl = F.kl_div(soft_student, soft_teacher, reduction="none").sum(dim=-1)  # (...,)

    if labels is not None:
        mask = (labels != ignore_index).float()
        return (kl * mask).sum() / mask.sum().clamp(min=1)

    return kl.mean()


def hidden_state_mse_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    projection: torch.nn.Linear | None = None,
) -> torch.Tensor:
    """
    Mean-squared-error loss between student and teacher hidden states.

    If the student and teacher have different hidden dimensions, pass a
    ``projection`` linear layer to project the student states before comparison.

    Args:
        student_hidden: (B, T, H_s) student hidden states.
        teacher_hidden: (B, T, H_t) teacher hidden states.
        attention_mask: (B, T) binary mask; 1 for real tokens, 0 for padding.
            If None, all positions are used.
        projection: Optional ``nn.Linear(H_s, H_t)`` applied to student before MSE.

    Returns:
        Scalar MSE loss, averaged over non-padding positions.
    """
    if projection is not None:
        student_hidden = projection(student_hidden)

    diff = (student_hidden - teacher_hidden) ** 2  # (B, T, H)

    if attention_mask is not None:
        mask = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
        return (diff * mask).sum() / (mask.sum() * diff.shape[-1]).clamp(min=1)

    return diff.mean()


def attention_map_mse_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean-squared-error loss between student and teacher attention maps.

    Attention tensors are expected to be *after* softmax (probability maps).
    When the student and teacher have different numbers of heads, the teacher
    maps are averaged over heads before comparison.

    Args:
        student_attention: (B, H_s, T, T) student attention probabilities.
        teacher_attention: (B, H_t, T, T) teacher attention probabilities.
        attention_mask: (B, T) binary mask for real tokens.  Positions where
            the key is padding are zeroed out in both maps before computing MSE.

    Returns:
        Scalar MSE loss.
    """
    # Align head counts by averaging teacher heads
    if student_attention.shape[1] != teacher_attention.shape[1]:
        teacher_attention = teacher_attention.mean(dim=1, keepdim=True).expand_as(
            student_attention.mean(dim=1, keepdim=True)
        )
        student_attention = student_attention.mean(dim=1, keepdim=True)

    if attention_mask is not None:
        # Zero out key-padding positions in both maps: (B, 1, 1, T)
        key_mask = attention_mask.float().unsqueeze(1).unsqueeze(2)
        student_attention = student_attention * key_mask
        teacher_attention = teacher_attention * key_mask

    return F.mse_loss(student_attention, teacher_attention)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    temperature: float,
    alpha: float,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined hard-label CE loss and soft-target KD loss.

    ``total = alpha * kd_loss + (1 - alpha) * ce_loss``

    When ``labels`` is None the CE term is omitted and ``alpha`` is treated as
    1.0 (pure KD loss).

    Args:
        student_logits: (B, T, V) or (B, V) student logits.
        teacher_logits: Same shape as student_logits.
        labels: (B, T) or (B,) hard labels.  Pass None for label-free distillation.
        temperature: Distillation temperature T.
        alpha: Interpolation weight (0 = pure CE, 1 = pure KD).
        ignore_index: Label value to ignore in CE loss.

    Returns:
        Tuple of (total_loss, ce_loss, kd_loss) scalars.
    """
    kd_loss = soft_target_kl_loss(
        student_logits, teacher_logits, temperature, ignore_index, labels
    )

    if labels is None:
        return kd_loss, torch.tensor(0.0, device=student_logits.device), kd_loss

    if student_logits.dim() == 3:
        B, T, V = student_logits.shape
        ce_loss = F.cross_entropy(
            student_logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=ignore_index,
        )
    else:
        ce_loss = F.cross_entropy(student_logits, labels, ignore_index=ignore_index)

    total = alpha * kd_loss + (1.0 - alpha) * ce_loss
    return total, ce_loss, kd_loss
