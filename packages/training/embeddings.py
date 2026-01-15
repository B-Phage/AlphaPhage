import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

ROPE_BASE = 10000.0


class RotaryPositionalEmbeddings(nn.Module):
    """
    Implements the RoPE mechanism for Query and Key tensors.
    The rotation is performed using complex multiplication for efficiency.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = ROPE_BASE):
        super().__init__()

        # RoPE applies to the per-head dimension
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 1. Calculate the inverse frequencies (theta_i) for each pair
        # Formula: theta_i = 1 / (base^(2*i / dim))
        # The torch.arange goes from 0 to dim-1, stepping by 2, covering d/2 pairs.
        theta_numerator = torch.arange(0, dim, 2).float()
        inv_freq = 1.0 / (base ** (theta_numerator / dim))  # (dim/2)

        # 2. Calculate position indices (t)
        t = torch.arange(max_seq_len, dtype=torch.float)  # (max_seq_len)

        # 3. Outer product: t * theta_i -> rotation angle matrix
        # Shape: (max_seq_len, dim/2) -> angle[pos, pair_index]
        freqs = torch.outer(t, inv_freq)

        # 4. Convert to complex numbers in polar form: R * exp(i*freqs)
        # R=1, angle=freqs. The result is cos(freqs) + i*sin(freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Register as a buffer for caching (not trainable)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor):  # , seq_len: int):
        """
        Applies the RoPE rotation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor (Query or Key) of shape
                              (B, H, T, D) or (B, T, D) or similar.
                              RoPE is applied to the last dimension (D).
            seq_len (int): The actual sequence length (T) of the input tensor.

        Returns:
            torch.Tensor: The rotated tensor.
        """
        # Select the correct number of position embeddings
        # freqs_cis = self.freqs_cis[:seq_len] # (T, D/2)
        freqs_cis = self.freqs_cis[: self.max_seq_len]  # (T, D/2) #new

        # 1. Reshape x to complex numbers (D -> D/2 pairs)
        # Shape: (..., T, D) -> (..., T, D/2, 2)
        # view_as_complex treats the last dim (size 2) as real/imag parts
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (..., T, D/2)

        # 2. Reshape freqs_cis for broadcasting
        # freqs_cis is (T, D/2), need to broadcast across batch (B) and heads (H)
        # Add necessary dimensions for broadcasting: (1, T, 1, D/2) or similar

        # freqs_cis is (T, D/2)
        # Reshape to (1, 1, T, D/2) to match (B, H, T, D/2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

        # 3. Apply the rotation via complex multiplication
        # The rotation formula is (a + ib) * (cos(theta) + i*sin(theta))
        x_rotated = x_complex * freqs_cis  # (..., T, D/2)

        # 4. Convert back to real numbers and flatten
        # view_as_real converts back to (..., T, D/2, 2)
        # flatten(3) converts (..., T, D/2, 2) back to (..., T, D)
        x_out = torch.view_as_real(x_rotated).flatten(-2).type_as(x)

        return x_out
