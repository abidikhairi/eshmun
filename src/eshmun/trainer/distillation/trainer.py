"""Knowledge distillation trainer."""

import logging
import os
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel

from eshmun.trainer.distillation.config import DistillationConfig
from eshmun.trainer.distillation.loggers import DistillationLogger
from eshmun.trainer.distillation.losses import (
    attention_map_mse_loss,
    distillation_loss,
    hidden_state_mse_loss,
)

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """
    Knowledge distillation trainer for HuggingFace PreTrainedModels.

    Trains a *student* model to match the output distribution of a frozen
    *teacher* model while simultaneously fitting the hard labels.  Optionally
    adds intermediate-layer supervision via hidden-state and attention-map MSE
    losses.

    The combined objective per step is:

        L = alpha * L_KD + (1 - alpha) * L_CE
          + hidden_loss_weight * L_hidden   (if distill_hidden_states)
          + attention_loss_weight * L_attn  (if distill_attentions)

    The teacher model is frozen at construction time (gradients disabled,
    eval mode).  The student model is trained normally.

    Dataset expectations:
        Each sample must yield a dict containing at minimum:
        - ``input_ids``  (LongTensor)
        - ``labels``     (LongTensor, with -100 at positions to ignore)
        Optional: ``attention_mask`` (LongTensor).

    Args:
        student: Student model to train.
        teacher: Teacher model (frozen).  Must produce the same output type as
            the student (same vocab size, hidden dimensionality for feature
            distillation).
        config: DistillationConfig with all hyperparameters.
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        data_collator: Collate function for the DataLoader.
        projections: Optional dict mapping ``"layer_{i}"`` keys to
            ``nn.Linear`` modules used to project student hidden states to the
            teacher hidden dimension before computing the hidden MSE loss.
        optimizers: Optional ``(optimizer, scheduler)`` tuple.  If either
            element is None the trainer creates a default AdamW + linear-warmup
            scheduler.
    """

    def __init__(
        self,
        student: PreTrainedModel,
        teacher: PreTrainedModel,
        config: DistillationConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        data_collator: Callable | None = None,
        projections: dict[str, nn.Linear] | None = None,
        optimizers: tuple = (None, None),
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.projections = projections or {}

        torch.manual_seed(config.seed)

        self.device = next(student.parameters()).device

        # Freeze the teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        ext_optimizer, ext_scheduler = optimizers
        self.optimizer = ext_optimizer or self._create_optimizer()
        self.scheduler = ext_scheduler or self._create_scheduler(self.optimizer)

        self._dtype = (
            torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else None
        )

        self._distill_logger = DistillationLogger(log_every=config.logging_steps)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full distillation training loop and return final metrics."""
        config = self.config
        student = self.student

        train_loader = self._get_dataloader(
            self.train_dataset, config.per_device_train_batch_size, shuffle=True
        )

        global_step = 0
        totals: dict[str, float] = {
            "total_loss": 0.0,
            "ce_loss": 0.0,
            "kd_loss": 0.0,
            "hidden_loss": 0.0,
            "attention_loss": 0.0,
        }

        for epoch in range(config.num_epochs):
            student.train()
            self.optimizer.zero_grad()

            for local_step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._dtype is not None,
                ):
                    step_loss, step_metrics = self._compute_loss(batch)
                    loss = step_loss / config.gradient_accumulation_steps

                loss.backward()

                for key, val in step_metrics.items():
                    totals[key] += val

                if local_step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    self._distill_logger.log(
                        global_step,
                        epoch + 1,
                        {
                            **step_metrics,
                            "lr": self.scheduler.get_last_lr()[0],
                        },
                    )

                    if config.save_steps > 0 and global_step % config.save_steps == 0:
                        self.save_model(
                            os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        )

                    if (
                        self.eval_dataset is not None
                        and config.eval_steps > 0
                        and global_step % config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        logger.info("eval step=%d %s", global_step, eval_metrics)
                        student.train()

            self._distill_logger.flush(epoch + 1)
            logger.info("epoch=%d complete", epoch + 1)

        steps_done = max(global_step, 1)
        return {f"train_{k}": v / steps_done for k, v in totals.items()}

    def evaluate(self) -> dict[str, float]:
        """Evaluate the student on eval_dataset and return metrics."""
        if self.eval_dataset is None:
            return {}

        student = self.student
        student.eval()

        eval_loader = self._get_dataloader(
            self.eval_dataset, self.config.per_device_eval_batch_size, shuffle=False
        )

        totals: dict[str, float] = {
            "total_loss": 0.0,
            "ce_loss": 0.0,
            "kd_loss": 0.0,
        }
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._dtype is not None,
                ):
                    _, step_metrics = self._compute_loss(batch)
                for k in ("total_loss", "ce_loss", "kd_loss"):
                    totals[k] += step_metrics.get(k, 0.0)

        n = len(eval_loader)
        return {f"eval_{k}": v / max(n, 1) for k, v in totals.items()}

    def save_model(self, output_dir: str | None = None) -> None:
        """Save student weights and config to output_dir."""
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.student.save_pretrained(save_dir)
        logger.info("student model saved to %s", save_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one forward pass and return total loss + metric dict."""
        config = self.config
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        need_hidden = config.distill_hidden_states
        need_attn = config.distill_attentions

        student_out = self.student(
            **{k: v for k, v in batch.items() if k != "labels"},
            output_hidden_states=need_hidden,
            output_attentions=need_attn,
        )

        with torch.no_grad():
            teacher_out = self.teacher(
                **{k: v for k, v in batch.items() if k != "labels"},
                output_hidden_states=need_hidden,
                output_attentions=need_attn,
            )

        total_loss, ce_loss, kd_loss = distillation_loss(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits,
            labels=labels,  # pyrefly: ignore [bad-argument-type]
            temperature=config.temperature,
            alpha=config.alpha,
        )

        metrics: dict[str, float] = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kd_loss": kd_loss.item(),
            "hidden_loss": 0.0,
            "attention_loss": 0.0,
        }

        if need_hidden and student_out.hidden_states and teacher_out.hidden_states:
            h_loss = self._hidden_loss(
                student_out.hidden_states,
                teacher_out.hidden_states,
                attention_mask,
            )
            weighted = config.hidden_loss_weight * h_loss
            total_loss = total_loss + weighted
            metrics["hidden_loss"] = weighted.item()

        if need_attn and student_out.attentions and teacher_out.attentions:
            a_loss = self._attention_loss(
                student_out.attentions,
                teacher_out.attentions,
                attention_mask,
            )
            weighted = config.attention_loss_weight * a_loss
            total_loss = total_loss + weighted
            metrics["attention_loss"] = weighted.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    def _hidden_loss(
        self,
        student_hidden: tuple[torch.Tensor, ...],
        teacher_hidden: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        layer_map = self.config.teacher_layer_map or list(range(len(student_hidden)))
        total = torch.tensor(0.0, device=self.device)
        for s_idx, t_idx in enumerate(layer_map):
            if s_idx >= len(student_hidden) or t_idx >= len(teacher_hidden):
                continue
            proj = self.projections.get(f"layer_{s_idx}")
            total = total + hidden_state_mse_loss(
                student_hidden[s_idx],
                teacher_hidden[t_idx],
                attention_mask=attention_mask,
                projection=proj,
            )
        return total / max(len(layer_map), 1)

    def _attention_loss(
        self,
        student_attentions: tuple[torch.Tensor, ...],
        teacher_attentions: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        layer_map = self.config.teacher_layer_map or list(range(len(student_attentions)))
        total = torch.tensor(0.0, device=self.device)
        for s_idx, t_idx in enumerate(layer_map):
            if s_idx >= len(student_attentions) or t_idx >= len(teacher_attentions):
                continue
            total = total + attention_map_mse_loss(
                student_attentions[s_idx],
                teacher_attentions[t_idx],
                attention_mask=attention_mask,
            )
        return total / max(len(layer_map), 1)

    def _create_optimizer(self) -> AdamW:
        no_decay = {"bias", "LayerNorm.weight"}
        params = [
            {
                "params": [
                    p for n, p in self.student.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.student.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.config.learning_rate)

    def _create_scheduler(self, optimizer: AdamW) -> LambdaLR:
        warmup = self.config.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _get_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            pin_memory=self.device.type == "cuda",
        )
