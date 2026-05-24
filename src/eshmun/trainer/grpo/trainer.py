"""GRPO (Group Relative Policy Optimization) trainer.

Implements the GRPO algorithm from the DeepSeek-R1 paper for aligning
decoder-only protein language models using reward functions.

Training loop (per batch):
  1. Generate G completions per prompt from the current policy.
  2. Score each (prompt, completion) pair with the reward functions.
  3. Compute group-relative advantages within each prompt's G completions.
  4. Run num_ppo_epochs gradient updates using the clipped policy-gradient
     objective and an approximate KL divergence penalty against a frozen
     reference model.
"""

import copy
import logging
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from eshmun.trainer.grpo.config import GRPOConfig

logger = logging.getLogger(__name__)

RewardFn = Callable[..., list[float]]


class GRPOTrainer:
    """
    GRPO trainer for decoder-only PreTrainedModels.

    The trainer maintains a frozen *reference model* (deep copy of the policy
    at init) used for the KL penalty term. The reference model can be synced
    to the current policy periodically via ``config.ref_model_sync_epochs``.

    Args:
        model: Decoder-only policy model to train. Must support ``.generate()``.
        config: GRPOConfig with training and GRPO hyperparameters.
        tokenizer: Tokenizer used to decode completions for reward scoring and
            to build padded prompt+completion batches.
        reward_fns: List of reward functions with signature
            ``(prompts: list[str], completions: list[str]) -> list[float]``.
            Rewards are combined via weighted sum (``config.reward_weights``).
        train_dataset: Dataset yielding dicts with ``input_ids`` and
            ``attention_mask`` (prompt tokens only).
        data_collator: Optional collate function for the DataLoader.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizerBase,
        reward_fns: list[RewardFn],
        train_dataset: Dataset,
        data_collator: Callable | None = None,
    ):
        if not reward_fns:
            raise ValueError("reward_fns must contain at least one reward function.")

        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.train_dataset = train_dataset
        self.data_collator = data_collator or self._left_pad_collator

        torch.manual_seed(config.seed)

        self.device = next(model.parameters()).device

        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        if config.reward_weights is not None:
            if len(config.reward_weights) != len(reward_fns):
                raise ValueError("reward_weights length must match reward_fns length.")
            w = torch.tensor(config.reward_weights, dtype=torch.float32)
        else:
            w = torch.ones(len(reward_fns), dtype=torch.float32)
        self._reward_weights = w / w.sum()

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

        self._dtype = (
            torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else None
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full GRPO training loop and return final metrics."""
        config = self.config
        model = self.model

        loader = DataLoader(
            self.train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            pin_memory=self.device.type == "cuda",
        )

        fn_names = [
            getattr(fn, "__name__", type(fn).__name__) for fn in self.reward_fns
        ]

        global_step = 0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        total_per_fn: list[float] = [0.0] * len(self.reward_fns)

        for epoch in range(config.num_epochs):
            for batch in loader:
                prompt_ids = batch["input_ids"].to(self.device)
                prompt_mask = batch.get(
                    "attention_mask",
                    torch.ones_like(prompt_ids),
                ).to(self.device)

                # --- Rollout phase (no grad) ---
                with torch.no_grad():
                    full_ids, completion_mask = self._generate_completions(
                        prompt_ids, prompt_mask
                    )
                    pad_id = int(self.tokenizer.pad_token_id or 0)
                    full_mask = (full_ids != pad_id).long()

                    log_probs_old = self._compute_per_token_log_probs(
                        model, full_ids, full_mask, completion_mask
                    )
                    log_probs_ref = self._compute_per_token_log_probs(
                        self.ref_model, full_ids, full_mask, completion_mask
                    )

                _tensor_keys = {"input_ids", "attention_mask"}
                batch_metadata = {k: v for k, v in batch.items() if k not in _tensor_keys} or None
                
                rewards, per_fn_means = self._compute_rewards(prompt_ids, full_ids, completion_mask, batch_metadata)
                advantages = self._compute_advantages(rewards, config.group_size)

                total_reward += rewards.mean().item()
                for i, m in enumerate(per_fn_means):
                    total_per_fn[i] += m

                # --- PPO update phase ---
                accum_steps = max(1, config.gradient_accumulation_steps)
                for _ in range(config.num_ppo_epochs):
                    model.train()
                    self.optimizer.zero_grad()

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self._dtype,
                        enabled=self._dtype is not None,
                    ):
                        log_probs_new = self._compute_per_token_log_probs(
                            model, full_ids, full_mask, completion_mask
                        )
                        policy_loss, kl = self._compute_grpo_loss(
                            log_probs_new,
                            log_probs_old,
                            log_probs_ref,
                            advantages,
                            completion_mask,
                        )
                        loss = (policy_loss + config.beta * kl) / accum_steps

                    loss.backward()

                    if (global_step + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        self.optimizer.step()
                        # MARKER: end of optimizer step here
                        self.optimizer.zero_grad()
                        self.scheduler.step()

                    total_policy_loss += policy_loss.item()
                    total_kl += kl.item()
                    global_step += 1

                    if global_step % config.logging_steps == 0:
                        per_fn_str = " ".join(
                            f"{name}=%.4f" for name in fn_names
                        )
                        logger.info(
                            f"epoch=%d step=%d policy_loss=%.4f kl=%.4f reward=%.4f {per_fn_str} lr=%.2e",
                            epoch + 1,
                            global_step,
                            policy_loss.item(),
                            kl.item(),
                            rewards.mean().item(),
                            *per_fn_means,
                            self.scheduler.get_last_lr()[0],
                        )

                    if config.save_steps > 0 and global_step % config.save_steps == 0:
                        self.save_model(
                            os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        )

            # Reference model sync
            if (
                config.ref_model_sync_epochs > 0
                and (epoch + 1) % config.ref_model_sync_epochs == 0
            ):
                self.ref_model.load_state_dict(model.state_dict())
                logger.info("reference model synced to policy at epoch %d", epoch + 1)

            logger.info("epoch=%d complete", epoch + 1)

        steps_done = max(global_step, 1)
        metrics: dict[str, float] = {
            "train_policy_loss": total_policy_loss / steps_done,
            "train_kl": total_kl / steps_done,
            "train_reward": total_reward / steps_done,
        }
        for name, total in zip(fn_names, total_per_fn):
            metrics[f"train_reward_{name}"] = total / steps_done
        return metrics

    def save_model(self, output_dir: str | None = None) -> None:
        """Save policy model weights and config to output_dir."""
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        logger.info("model saved to %s", save_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_completions(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate G completions per prompt via sampling.

        Args:
            prompt_ids: (B, T_p) prompt token ids.
            prompt_mask: (B, T_p) attention mask for prompt tokens.

        Returns:
            full_ids: (B*G, T_p + T_c) prompt + completion token ids.
            completion_mask: (B*G, T_p + T_c) 1 for completion positions, 0 elsewhere.
        """
        config = self.config
        B, T_p = prompt_ids.shape
        G = config.group_size

        # Repeat each prompt G times: (B*G, T_p)
        repeated_ids = prompt_ids.repeat_interleave(G, dim=0)
        repeated_mask = prompt_mask.repeat_interleave(G, dim=0)

        with torch.no_grad():
            output_ids = self.model.generate(  # pyrefly: ignore [not-callable]
                input_ids=repeated_ids,
                attention_mask=repeated_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=50266 # NOTE: token '</protein>'
            )

        T_full = output_ids.shape[1]

        # Completion positions are everything after the original prompt tokens.
        # Positions before T_p are prompt; T_p onwards are completion (until pad).
        completion_mask = torch.zeros(B * G, T_full, dtype=torch.long, device=self.device)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        completion_mask[:, T_p:] = (output_ids[:, T_p:] != pad_id).long()

        return output_ids, completion_mask

    def _compute_per_token_log_probs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for completion tokens.

        Args:
            model: Model to use for forward pass.
            input_ids: (B*G, T) full sequence token ids.
            attention_mask: (B*G, T) attention mask.
            completion_mask: (B*G, T) 1 for completion token positions.

        Returns:
            log_probs: (B*G, T-1) per-token log probabilities, zeroed at non-completion positions.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B*G, T, V)

        # Shift: predict token t+1 using logits at position t
        shift_logits = logits[:, :-1, :]         # (B*G, T-1, V)
        shift_labels = input_ids[:, 1:]           # (B*G, T-1)
        shift_completion = completion_mask[:, 1:] # (B*G, T-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B*G, T-1)

        return token_log_probs * shift_completion
        # MARKER: end of forward call here

    def _compute_rewards(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        batch_metadata: dict[str, list[Any]] | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Decode prompts and completions, apply reward functions, return weighted sum.

        Args:
            batch_metadata: Optional dict of extra per-sample data from the dataset
                (e.g. ``{"protein_family_stat": [...]}``) keyed by column name.
                Each value is a list of length B (one entry per prompt).  The trainer
                repeats every entry G times so the expanded list aligns with the B*G
                (prompt, completion) pairs before forwarding as ``**kwargs`` to every
                reward function.  Reward functions that do not declare the kwarg simply
                ignore it.

        Returns:
            rewards: (B*G,) weighted combined reward tensor.
            per_fn_means: mean score for each reward function, in reward_fns order.
        """
        B_times_G = full_ids.shape[0]
        T_p = prompt_ids.shape[1]
        G = self.config.group_size

        prompt_ids_repeated = prompt_ids.repeat_interleave(G, dim=0)
        prompt_strs = self.tokenizer.batch_decode(
            prompt_ids_repeated, skip_special_tokens=False
        )

        # Extract only the completion token ids for decoding
        pad_id = int(self.tokenizer.pad_token_id or 0)
        completion_ids = full_ids.clone()
        completion_ids[completion_mask == 0] = pad_id
        completion_strs = self.tokenizer.batch_decode(
            completion_ids[:, T_p:], skip_special_tokens=False
        )
        # MARKER: end of formatting output here

        # Expand metadata from B entries to B*G to align with completions
        expanded_meta: dict[str, list[Any]] = {}
        if batch_metadata:
            for key, values in batch_metadata.items():
                expanded_meta[key] = [v for v in values for _ in range(G)]

        weights = self._reward_weights.to(self.device)
        rewards = torch.zeros(B_times_G, device=self.device)
        per_fn_means: list[float] = []

        for i, fn in enumerate(self.reward_fns):            
            scores = torch.tensor(
                fn(prompt_strs, completion_strs, **expanded_meta),
                dtype=torch.float32,
                device=self.device,
            )

            # MARKER: end of calling reward functions here
            per_fn_means.append(scores.mean().item())
            rewards += weights[i] * scores
        return rewards, per_fn_means

    def _compute_advantages(
        self, rewards: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        """
        Group-normalize rewards to produce advantages.

        For each prompt's group of G completions:
            A_i = (r_i - mean_g) / (std_g + 1e-8)

        Args:
            rewards: (B*G,) flat rewards.
            group_size: G completions per prompt.

        Returns:
            advantages: (B*G,) group-normalized advantages.
        """
        rewards = rewards.view(-1, group_size)   # (B, G)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, correction=0)
        advantages = (rewards - mean) / (std + 1e-8)
        return advantages.view(-1)               # (B*G,)

    def _compute_grpo_loss(
        self,
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_ref: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the clipped policy-gradient loss and approximate KL divergence.

        Policy gradient (per token):
            ratio_t = exp(log_π_new_t - log_π_old_t)
            loss_t  = -min(ratio_t * A, clip(ratio_t, 1-ε, 1+ε) * A)

        Approximate unbiased KL (Schulman et al.):
            kl_t = exp(log_π_ref_t - log_π_new_t) - (log_π_ref_t - log_π_new_t) - 1

        Both are averaged over completion tokens only.

        Args:
            log_probs_new: (B*G, T-1) current policy log probs.
            log_probs_old: (B*G, T-1) old policy log probs (no grad).
            log_probs_ref: (B*G, T-1) reference model log probs (no grad).
            advantages: (B*G,) group-normalized advantages.
            completion_mask: (B*G, T) completion indicator; shifted by 1 inside.

        Returns:
            policy_loss: scalar
            kl: scalar
        """
        eps = self.config.epsilon
        shift_mask = completion_mask[:, 1:].float()  # (B*G, T-1)
        n_tokens = shift_mask.sum().clamp(min=1)

        ratio = torch.exp(log_probs_new - log_probs_old)           # (B*G, T-1)
        A = advantages.unsqueeze(1)                                  # (B*G, 1)

        pg_loss = -torch.min(
            ratio * A,
            torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * A,
        )
        policy_loss = (pg_loss * shift_mask).sum() / n_tokens

        log_diff = log_probs_ref - log_probs_new
        kl_per_token = torch.exp(log_diff) - log_diff - 1.0
        kl = (kl_per_token * shift_mask).sum() / n_tokens

        return policy_loss, kl

    def _left_pad_collator(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Left-pad prompt sequences to the same length within a batch.

        Left-padding keeps the last real token at the rightmost position so the
        model generates naturally from there.  Right-padding would force generation
        after the pad tokens, producing garbage.
        """
        pad_id = int(self.tokenizer.pad_token_id or 0)
        input_ids = [item["input_ids"] for item in batch]
        masks = [
            item.get("attention_mask", torch.ones_like(item["input_ids"]))
            for item in batch
        ]
        max_len = max(t.shape[0] for t in input_ids)

        padded_ids, padded_masks = [], []
        for ids, mask in zip(input_ids, masks):
            pad_len = max_len - ids.shape[0]
            padded_ids.append(
                torch.cat([ids.new_full((pad_len,), pad_id), ids])
            )
            padded_masks.append(
                torch.cat([mask.new_zeros(pad_len), mask])
            )

        result: dict[str, Any] = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_masks),
        }
        for key in batch[0]:
            if key not in ("input_ids", "attention_mask"):
                result[key] = [item[key] for item in batch]

        return result

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
