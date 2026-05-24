"""Configuration for supervised fine-tuning."""

from dataclasses import dataclass, field


@dataclass
class SFTConfig:
    """
    Training configuration for SFTTrainer.

    Args:
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device for training.
        per_device_eval_batch_size: Batch size per device for evaluation.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: Weight decay coefficient for AdamW.
        warmup_steps: Number of linear warmup steps before constant LR.
        gradient_accumulation_steps: Accumulate gradients over N steps before updating.
        max_grad_norm: Max gradient norm for clipping.
        logging_steps: Log training metrics every N optimizer steps.
        save_steps: Save a checkpoint every N optimizer steps.
        eval_steps: Run evaluation every N optimizer steps.
        bf16: Use bfloat16 mixed precision.
        fp16: Use float16 mixed precision.
        seed: Random seed.
    """

    output_dir: str
    num_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    bf16: bool = False
    fp16: bool = False
    seed: int = 42
