"""Molecular reward functions (drug design tasks).

Requires rdkit: `pip install rdkit`
All classes are callable and satisfy the RewardFn signature:
    (prompts: Sequence[str], completions: Sequence[str]) -> list[float]

Functions degrade gracefully when rdkit is not installed: they raise
ImportError at instantiation time so the failure is visible immediately
rather than silently at scoring time.
"""
from collections.abc import Sequence

from eshmun.trainer.grpo.reward_functions.registry import register


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401  # pyrefly: ignore [missing-import]
    except ImportError as e:
        raise ImportError(
            "rdkit is required for molecular reward functions. "
            "Install it with: pip install rdkit"
        ) from e


@register
class SMILESValidityReward:
    """
    Returns 1.0 if the completion is a valid SMILES string, 0.0 otherwise.

    Completions are stripped of whitespace before parsing. Requires rdkit.
    """

    def __init__(self):
        _require_rdkit()
        from rdkit import Chem  # pyrefly: ignore [missing-import]
        self._Chem = Chem

    def __call__(self, prompts: Sequence[str], completions: Sequence[str]) -> list[float]:
        scores = []
        for smi in completions:
            mol = self._Chem.MolFromSmiles(smi.strip())
            scores.append(1.0 if mol is not None else 0.0)
        return scores


@register
class QEDReward:
    """
    Returns the Quantitative Estimate of Drug-likeness (QED) score for each
    completion, interpreted as a SMILES string. Invalid SMILES receive 0.0.

    QED ranges from 0 (least drug-like) to 1 (most drug-like). Requires rdkit.
    """

    def __init__(self):
        _require_rdkit()
        from rdkit import Chem  # pyrefly: ignore [missing-import]
        from rdkit.Chem import QED  # pyrefly: ignore [missing-import]
        self._Chem = Chem
        self._QED = QED

    def __call__(self, prompts: Sequence[str], completions: Sequence[str]) -> list[float]:
        scores = []
        for smi in completions:
            mol = self._Chem.MolFromSmiles(smi.strip())
            if mol is None:
                scores.append(0.0)
            else:
                try:
                    scores.append(float(self._QED.qed(mol)))
                except Exception:
                    scores.append(0.0)
        return scores


@register
class SAScoreReward:
    """
    Returns a normalized Synthetic Accessibility (SA) score reward.

    SA score ranges from 1 (easy to synthesize) to 10 (hard). This reward
    maps it to [0, 1] where 1.0 = easiest to synthesize:

        reward = (10 - sa_score) / 9

    Invalid SMILES receive 0.0. Requires rdkit and sascorer.
    """

    def __init__(self):
        _require_rdkit()
        try:
            from rdkit.Contrib.SA_Score import sascorer  # pyrefly: ignore [missing-import]
            self._sascorer = sascorer
        except ImportError as e:
            raise ImportError(
                "sascorer not found. Make sure rdkit is installed with Contrib support."
            ) from e
        from rdkit import Chem  # pyrefly: ignore [missing-import]
        self._Chem = Chem

    def __call__(self, prompts: Sequence[str], completions: Sequence[str]) -> list[float]:
        scores = []
        for smi in completions:
            mol = self._Chem.MolFromSmiles(smi.strip())
            if mol is None:
                scores.append(0.0)
            else:
                try:
                    sa = self._sascorer.calculateScore(mol)
                    scores.append((10.0 - sa) / 9.0)
                except Exception:
                    scores.append(0.0)
        return scores
