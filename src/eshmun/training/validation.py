"""Recipe-key validation: a typo'd key in a recipe is otherwise swallowed.

`TrainingArguments` accepts arbitrary extra kwargs, so an unknown key would
train with the value it should have had nowhere in sight. Validation runs
before anything is loaded, so a bad recipe fails cheaply.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from eshmun.errors import fail

# Names that older transformers/TRL accepted, mapped to the name this
# environment wants. Only renames whose value carries over unchanged belong
# here, so that pointing at the new name is always safe.
RENAMED_KEYS = {
    "evaluation_strategy": "eval_strategy",
    # transformers 5.x dropped `warmup_ratio`; `warmup_steps` now takes a float
    # below 1 as a fraction of total steps, which is what a ratio was.
    "warmup_ratio": "warmup_steps",
}


def validate_keys(section: str, given: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(given) - known)
    if not unknown:
        return

    hints = []
    for key in unknown:
        renamed = RENAMED_KEYS.get(key)
        close = [k for k in known if key.lower() in k.lower() or k.lower() in key.lower()]
        if renamed and renamed in known:
            hints.append(f"  - {key} (renamed: use `{renamed}`)")
        elif close:
            hints.append(f"  - {key} (did you mean: {', '.join(sorted(close))}?)")
        else:
            hints.append(f"  - {key}")
    fail(f"unknown key(s) in recipe section `{section}`:\n" + "\n".join(hints))


def validate_section(section: str, given: dict[str, Any], target: type) -> None:
    """Reject keys that the destination dataclass would silently swallow.

    `TrainingArguments` accepts arbitrary extra kwargs, so a typo like
    `learning_rates` would otherwise be ignored without a word.
    """
    validate_keys(section, given, {f.name for f in dataclasses.fields(target)})
