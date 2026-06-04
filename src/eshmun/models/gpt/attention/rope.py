from typing import Tuple
import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        i = torch.arange(0, head_dim, 2).float()
        theta = 1.0 / (base ** (i / head_dim))

        positions = torch.arange(max_seq_len).float()

        angles = torch.outer(positions, theta)
        
        self.register_buffer("cos_table", angles.cos())
        self.register_buffer("sin_table", angles.sin())

    def apply_rotation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-2]
        
        # pyrefly: ignore [bad-index]
        cos_table = self.cos_table[:L, :]
        # pyrefly: ignore [bad-index]
        sin_table = self.sin_table[:L, :]
        
        x1 = x[..., 0::2]  # even positions
        x2 = x[..., 1::2]  # odd positions

        x1_rotated = x1 * cos_table - x2 * sin_table
        x2_rotated = x1 * sin_table + x2 * cos_table

        return torch.stack([x1_rotated, x2_rotated], dim=-1).flatten(-2)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_rotated = self.apply_rotation_matrix(q)
        k_rotated = self.apply_rotation_matrix(k)

        return (q_rotated, k_rotated)
