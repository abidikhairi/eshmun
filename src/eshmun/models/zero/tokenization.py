"""
EshmunTokenizer
===============
Residue-level protein tokenizer wrapping HuggingFace PreTrainedTokenizer.

Vocabulary (30 tokens):
  - 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY)
  - 5 IUPAC ambiguity codes (B, Z, X, U, O)
  - 5 special tokens: <pad>, <bos>, <eos>, <unk>, <mask>

Design decisions:
  - Character-level: one token per residue, no subword compression.
  - Case-insensitive input: sequences are uppercased before encoding.
  - Left-padding by default (matches causal LM convention for batching).
  - Implements the full PreTrainedTokenizer interface so it works with
    HuggingFace Trainer, DataCollatorForLanguageModeling, etc.

Usage:
    tokenizer = EshmunTokenizer()
    tokenizer.save_pretrained("./eshmun-tokenizer")

    enc = tokenizer("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPsvmenpqkvlelhlanqlkpievvqnlrgeqygmpnmftesvdlagnaesgstgeqilqkldvfkqnhfmsiiddqttrkkqqqqqqppptttttqqhgkeqshkmaerleaaeqhkkerkqqhheqkliseedlmkqnstvsmpqkliseedlsqnstvsmpqkliseedlaqkkliseedlsqnstvsmpqkliseedlsqnstvsmpqkli")
    print(enc.input_ids)      # [2, 12, 15, ...]
    print(enc.attention_mask) # [1, 1, 1, ...]

    # Batch with padding
    batch = tokenizer(
        ["MKTAY", "ACDEFGHIKLM"],
        padding=True,
        return_tensors="pt",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer

STANDARD_AAS: str = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
IUPAC_EXTRA: str = "BZOUXJ"  # ambiguity + selenocysteine + pyrrolysine + unknown-rare

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

# Canonical token order: special first, then AA alphabet
_SPECIAL_LIST = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
_AA_LIST = list(STANDARD_AAS + IUPAC_EXTRA)  # 26 residue tokens
_VOCAB_LIST = _SPECIAL_LIST + _AA_LIST  # 31 tokens total


def _build_vocab() -> Dict[str, int]:
    return {tok: idx for idx, tok in enumerate(_VOCAB_LIST)}


class EshmunZeroTokenizer(PreTrainedTokenizer):
    """
    Residue-level tokenizer for EshmunGPT protein language models.

    Each amino acid character maps to exactly one token ID.
    Unknown characters map to <unk>. Input is always uppercased.

    Special token IDs (fixed):
        <pad>  = 0
        <bos>  = 1
        <eos>  = 2
        <unk>  = 3
        <mask> = 4
        A      = 5
        C      = 6
        ...
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        **kwargs,
    ):
        self._vocab: Dict[str, int] = _build_vocab()
        self._ids_to_tokens: Dict[int, str] = {v: k for k, v in self._vocab.items()}

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

    # ── Core vocabulary interface ──────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of tokens in vocabulary (31: 5 special + 26 residue)."""
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Split a protein sequence string into a list of single-character tokens.
        Input is uppercased; unknown characters become '<unk>'.
        """
        text = text.upper().strip()
        tokens = []
        for ch in text:
            if ch in self._vocab:
                tokens.append(ch)
            else:
                tokens.append(self.unk_token)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Reconstruct a sequence string from a list of tokens.
        Special tokens are dropped; residue tokens are joined directly.
        """
        special = set(SPECIAL_TOKENS.values())
        return "".join(t for t in tokens if t not in special)

    # ── Special token handling ─────────────────────────────────────────────────

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Add <bos> at the start and <eos> at the end.
        Protein sequences are always single-segment; token_ids_1 is unused.
        """
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        return bos + token_ids_0 + eos

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return [
                1 if self._ids_to_tokens.get(t, "") in SPECIAL_TOKENS.values() else 0
                for t in token_ids_0
            ]
        bos_mask = [1] if self.add_bos_token else []
        eos_mask = [1] if self.add_eos_token else []
        return bos_mask + [0] * len(token_ids_0) + eos_mask

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """All tokens belong to segment 0."""
        bos = [0] if self.add_bos_token else []
        eos = [0] if self.add_eos_token else []
        return bos + [0] * len(token_ids_0) + eos

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """Save vocab.json to save_directory."""
        # pyrefly: ignore [bad-assignment]
        save_directory_path = Path(save_directory)
        save_directory_path.mkdir(parents=True, exist_ok=True)

        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_file = save_directory_path / f"{prefix}vocab.json"

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, indent=2, ensure_ascii=False)

        return (str(vocab_file),)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load from a directory saved with save_pretrained().
        The vocab is always rebuilt from the canonical constant —
        the saved vocab.json is only checked for consistency.
        """
        vocab_file = Path(pretrained_model_name_or_path) / "vocab.json"
        if vocab_file.exists():
            with open(vocab_file, encoding="utf-8") as f:
                saved_vocab = json.load(f)
            expected = _build_vocab()
            if saved_vocab != expected:
                import warnings

                warnings.warn(
                    "Saved vocab.json differs from canonical EshmunTokenizer vocab. "
                    "Using canonical vocab. Re-save to suppress this warning."
                )
        return cls(*args, **kwargs)

    # ── Convenience helpers ────────────────────────────────────────────────────

    def aa_to_id(self, aa: str) -> int:
        """Single amino acid character → token ID."""
        return self._vocab.get(aa.upper(), self.unk_token_id)

    def id_to_aa(self, idx: int) -> str:
        """Token ID → amino acid character (or special token string)."""
        return self._ids_to_tokens.get(idx, self.unk_token)

    def decode_sequence(
        self, token_ids: List[int], skip_special_tokens: bool = True
    ) -> list[str] | str:
        """
        Decode a list of token IDs back to a protein sequence string.
        By default strips <bos>, <eos>, <pad>, <mask>, <unk>.
        """
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def standard_aa_ids(self) -> List[int]:
        """Token IDs for the 20 standard amino acids only."""
        return [self._vocab[aa] for aa in STANDARD_AAS]

    @property
    def all_aa_ids(self) -> List[int]:
        """Token IDs for all 26 residue tokens (standard + IUPAC ambiguity)."""
        return [self._vocab[aa] for aa in _AA_LIST]

    def __repr__(self) -> str:
        return (
            f"EshmunTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"add_bos={self.add_bos_token}, "
            f"add_eos={self.add_eos_token})"
        )
