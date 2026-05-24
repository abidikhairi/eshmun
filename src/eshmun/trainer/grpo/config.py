"""Configuration for GRPO training."""

from dataclasses import dataclass

from eshmun.trainer.sft.config import SFTConfig


@dataclass
class GRPOConfig(SFTConfig):
    """
    Training configuration for GRPOTrainer. Extends SFTConfig with GRPO-specific fields.

    Args:
        group_size: Number of completions sampled per prompt (G in the GRPO paper).
        epsilon: PPO clipping coefficient for the policy ratio.
        beta: KL divergence penalty coefficient.
        num_ppo_epochs: Number of gradient update passes per rollout batch.
        temperature: Sampling temperature for completion generation.
        max_new_tokens: Maximum number of tokens to generate per completion.
        reward_weights: Per-reward-function weights for weighted sum. Uniform if None.
        ref_model_sync_epochs: Sync the reference model to the current policy every N
            training epochs. Set to 0 to keep the reference model fixed for the full run.
    """

    group_size: int = 8
    epsilon: float = 0.2
    beta: float = 0.01
    num_ppo_epochs: int = 1
    temperature: float = 1.0
    max_new_tokens: int = 512
    reward_weights: list[float] | None = None
    ref_model_sync_epochs: int = 1
