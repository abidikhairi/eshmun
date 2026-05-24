"""Reward function registry.

Classes decorated with ``@register`` are stored here and can be looked up
by name when building the reward pipeline from a config file.
"""

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Register a reward function class under its ``__name__``."""
    _REGISTRY[cls.__name__] = cls
    return cls


def get(name: str) -> type:
    """Return the class registered under *name*, raising KeyError if missing."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown reward function {name!r}. Available: {available}"
        )
    return _REGISTRY[name]


def list_all() -> dict[str, str]:
    """Return ``{name: docstring}`` for every registered reward function."""
    return {
        name: (cls.__doc__ or "").strip()
        for name, cls in sorted(_REGISTRY.items())
    }
