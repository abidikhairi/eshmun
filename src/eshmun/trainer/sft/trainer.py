"""Supervised fine-tuning trainer."""

import logging
import math
import os
from typing import Callable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel

from eshmun.trainer.sft.config import SFTConfig

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    Supervised fine-tuning trainer for any HuggingFace PreTrainedModel.

    Expects the dataset to yield dicts with at minimum ``input_ids`` and
    ``labels``. A ``data_collator`` is applied inside the DataLoader and is
    responsible for padding and any dynamic masking (e.g. MLM).

    Args:
        model: The model to fine-tune.
        config: SFTConfig with training hyperparameters.
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        data_collator: Callable applied as the DataLoader collate function.
        optimizers: Optional ``(optimizer, scheduler)`` tuple. When either
            element is None the trainer creates a default AdamW optimizer and
            a linear-warmup + constant scheduler.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: SFTConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        data_collator: Callable | None = None,
        optimizers: tuple = (None, None),
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        torch.manual_seed(config.seed)

        self.device = next(model.parameters()).device

        ext_optimizer, ext_scheduler = optimizers
        self.optimizer = ext_optimizer or self._create_optimizer()
        self.scheduler = ext_scheduler or self._create_scheduler(self.optimizer)

        self._dtype = (
            torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else None
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full training loop and return final training metrics."""
        config = self.config
        model = self.model
        model.train()

        train_loader = self._get_dataloader(
            self.train_dataset, config.per_device_train_batch_size, shuffle=True
        )

        global_step = 0
        total_loss = 0.0
        log_loss = 0.0

        for epoch in range(config.num_epochs):
            self.optimizer.zero_grad()

            for local_step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._dtype is not None,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps

                loss.backward()
                total_loss += loss.item() * config.gradient_accumulation_steps
                log_loss += loss.item() * config.gradient_accumulation_steps

                if local_step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % config.logging_steps == 0:
                        avg = log_loss / config.logging_steps
                        logger.info(
                            "epoch=%d step=%d loss=%.4f lr=%.2e",
                            epoch + 1,
                            global_step,
                            avg,
                            self.scheduler.get_last_lr()[0],
                        )
                        log_loss = 0.0

                    if config.save_steps > 0 and global_step % config.save_steps == 0:
                        self.save_model(os.path.join(config.output_dir, f"checkpoint-{global_step}"))

                    if (
                        self.eval_dataset is not None
                        and config.eval_steps > 0
                        and global_step % config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        logger.info("eval step=%d %s", global_step, eval_metrics)
                        model.train()

            logger.info("epoch=%d complete", epoch + 1)

        steps_done = max(global_step, 1)
        return {"train_loss": total_loss / steps_done}

    def evaluate(self) -> dict[str, float]:
        """Evaluate on eval_dataset and return metrics."""
        if self.eval_dataset is None:
            return {}

        model = self.model
        model.eval()

        eval_loader = self._get_dataloader(
            self.eval_dataset, self.config.per_device_eval_batch_size, shuffle=False
        )

        total_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._dtype is not None,
                ):
                    outputs = model(**batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(eval_loader)
        return {"eval_loss": avg_loss, "eval_perplexity": math.exp(avg_loss)}

    def save_model(self, output_dir: str | None = None) -> None:
        """Save model weights and config to output_dir."""
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        logger.info("model saved to %s", save_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_optimizer(self) -> AdamW:
        no_decay = {"bias", "LayerNorm.weight"}
        params = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
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

    def _get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            pin_memory=self.device.type == "cuda",
        )
