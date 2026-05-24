# GRPO Training

GRPO (Group Relative Policy Optimization) fine-tunes the Eshmun decoder model using reward functions instead of labeled data. It follows the algorithm from the DeepSeek-R1 paper: for each prompt, G completions are sampled, scored by reward functions, normalized within the group to produce advantages, then used to update the policy with a clipped PPO objective and a KL penalty against a frozen reference model.

## Prerequisites

```bash
pip install -e .
pip install pyyaml datasets transformers
# For SequenceSimilarityReward only:
pip install biopython
```

## Quick start

```bash
# 1. Copy and edit the config
cp scripts/grpo/config.yaml my_run.yaml
# (edit model, dataset, rewards, etc.)

# 2. Run training
python scripts/grpo/train.py train --config my_run.yaml

# 3. List available reward functions
python scripts/grpo/train.py list-rewards
```

## Config file reference

The config is a YAML file with five top-level sections.

### `model`

```yaml
model:
  name_or_path: "facebook/opt-125m"   # HuggingFace ID or local path
```

Any `AutoModelForCausalLM`-compatible model works. Use a local Eshmun checkpoint path for Eshmun-specific fine-tuning.

### `tokenizer`

```yaml
tokenizer:
  name_or_path: "facebook/opt-125m"   # can differ from model (rare)
```

Omit to reuse `model.name_or_path`. The script sets `pad_token_id = eos_token_id` automatically when the tokenizer has no pad token.

### `dataset`

```yaml
dataset:
  name_or_path: "tatsu-lab/alpaca"  # HuggingFace dataset ID or local path
  split: "train"
  prompt_column: "instruction"      # column containing the prompt strings
  max_prompt_length: 512            # truncate prompts to this many tokens

  chat_template:
    enabled: false          # true for instruct-tuned models with a chat template
    system_prompt: null     # optional system prompt prepended to every user turn
```

The dataset must have at least one text column for prompts. The script tokenizes it and discards all other columns. If `chat_template.enabled` is `true` but the tokenizer has no `chat_template`, the script falls back to plain tokenization with a warning.

### `training`

```yaml
training:
  output_dir: "/tmp/outputs/grpo-run"

  # General
  num_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-5
  weight_decay: 0.01
  warmup_steps: 20
  max_grad_norm: 1.0
  seed: 42

  # Logging / checkpointing
  logging_steps: 10     # log metrics every N optimizer steps
  save_steps: 200       # save checkpoint every N optimizer steps (0 = disabled)
  eval_steps: 200       # (reserved for future eval integration)

  # Mixed precision — set at most one to true
  bf16: false
  fp16: false

  # GRPO-specific
  group_size: 4           # G: completions sampled per prompt
  epsilon: 0.2            # PPO clipping coefficient
  beta: 0.01              # KL penalty coefficient
  num_ppo_epochs: 1       # gradient updates per rollout batch
  temperature: 1.0        # sampling temperature during rollout
  max_new_tokens: 128     # max completion tokens
  ref_model_sync_epochs: 1  # sync reference model every N epochs (0 = never)
```

**Effective batch size** = `per_device_train_batch_size × gradient_accumulation_steps × group_size`.

**Reference model sync**: when `ref_model_sync_epochs > 0`, the frozen reference model is updated to the current policy weights every N epochs, gradually relaxing the KL constraint. Set to `0` to keep it fixed for the entire run.

### `rewards`

```yaml
rewards:
  - name: SequenceValidityReward
  - name: LengthReward
    kwargs:
      min_length: 50
      max_length: 500
      out_of_range_penalty: -1.0

reward_weights: [0.4, 0.6]   # must match length of rewards; null for uniform
```

The final reward for each completion is the weighted sum of all reward function scores. `reward_weights` is normalized to sum to 1 internally.

## Built-in reward functions

| Name | Returns | Notes |
|---|---|---|
| `SequenceValidityReward` | `1.0` / `0.0` | All chars must be standard amino acids (A–Y). Sequence must be wrapped in `<protein>…</protein>`. |
| `LengthReward` | `1.0` / `out_of_range_penalty` | Passes if `min_length ≤ len(seq) ≤ max_length`. Default penalty `0.0`. |
| `CompositeProteinReward` | `1.0` / `0.0` | `SequenceValidityReward × LengthReward`. Fails both checks if either fails. |
| `SequenceSimilarityReward` | `[0, 1]` | Global pairwise identity against a reference sequence. Requires `biopython`. Returns `non_valid_seq_reward` (default `-1`) when parsing fails. |

Use `python scripts/grpo/train.py list-rewards` to see the live registry with descriptions.

## Writing a custom reward function

Implement the `RewardFn` signature and register with `@register`:

```python
from eshmun.trainer.grpo.reward_functions.registry import register

@register
class MyReward:
    """One-line description shown by list-rewards."""

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        return [1.0 if "ACGT" in c else 0.0 for c in completions]
```

Import the module before training so the decorator runs and the class appears in the registry.

## Sequence format

Eshmun completions wrap the protein sequence in XML-like tags:

```
<protein>MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL</protein>
```

`SequenceValidityReward` and `LengthReward` both call `extract_sequence()` to parse this tag. Custom reward functions should do the same:

```python
from eshmun.trainer.grpo.reward_functions.protein import extract_sequence
```

## Output

Checkpoints are saved to `output_dir/checkpoint-{step}/` every `save_steps` steps. The final model is saved to `output_dir/` at the end of training. Both use `model.save_pretrained()` and are loadable with `AutoModelForCausalLM.from_pretrained`.

Training logs include:

```
12:34:56 INFO eshmun.grpo.train — epoch=1 step=10 policy_loss=0.1234 kl=0.0056 reward=0.4200 lr=1.00e-05
```

The `train()` method returns `{"train_policy_loss": ..., "train_kl": ..., "train_reward": ...}` averaged over all steps.

## Programmatic usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from eshmun.trainer.grpo import GRPOConfig, GRPOTrainer
from eshmun.trainer.grpo.reward_functions.protein import SequenceValidityReward, LengthReward

model = AutoModelForCausalLM.from_pretrained("my-eshmun-checkpoint")
tokenizer = AutoTokenizer.from_pretrained("my-eshmun-checkpoint")
tokenizer.pad_token_id = tokenizer.eos_token_id

config = GRPOConfig(
    output_dir="outputs/grpo",
    num_epochs=3,
    per_device_train_batch_size=2,
    group_size=8,
    beta=0.01,
    epsilon=0.2,
    temperature=1.0,
    max_new_tokens=256,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    reward_fns=[SequenceValidityReward(), LengthReward(min_length=50, max_length=512)],
    train_dataset=my_dataset,  # yields dicts with input_ids and attention_mask
)

metrics = trainer.train()
trainer.save_model()
```

## Troubleshooting

**OOM during rollout**: reduce `group_size` or `max_new_tokens`. Rollout runs under `torch.no_grad()` but still allocates G × sequence-length tensors.

**All rewards are 0**: check that completions contain `<protein>…</protein>` tags. Run `list-rewards` to confirm reward functions are loaded.

**`reward_weights` mismatch**: the list length must equal the number of entries in `rewards`. The script raises `ValueError` at startup.

**Tokenizer has no chat template**: set `chat_template.enabled: false` or use a model with a bundled chat template (e.g. Llama-3-Instruct).
