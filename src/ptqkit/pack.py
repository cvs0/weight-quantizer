"""Pack unsigned 4-bit codes into bytes.

Layout, row-major over the input dimension K:

    packed[m, j] = (q[m, 2j] & 15) | ((q[m, 2j + 1] & 15) << 4)

Index 2j is the low nibble. Index 2j+1 is the high nibble. An odd K is padded
with a zero code so the byte array is rectangular. The pad nibble is not part
of the logical row and must not be consumed by the matmul.
"""

from __future__ import annotations

import torch


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack integer codes of shape [M, K] into uint8 of shape [M, ceil(K/2)]."""
    if codes.ndim != 2:
        raise ValueError("codes must have shape [M, K]")
    q = codes.to(dtype=torch.int32)
    if q.numel() > 0:
        lo = int(q.min().item())
        hi = int(q.max().item())
        if lo < 0 or hi > 15:
            raise ValueError(f"int4 codes must lie in [0, 15], got [{lo}, {hi}]")
    if q.shape[1] % 2 == 1:
        pad = torch.zeros(q.shape[0], 1, dtype=torch.int32, device=q.device)
        q = torch.cat((q, pad), dim=1)
    low = q[:, 0::2] & 0x0F
    high = q[:, 1::2] & 0x0F
    packed = (low | (high << 4)).to(dtype=torch.uint8)
    return packed.contiguous()


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Invert `pack_int4` and drop the pad nibble past `in_features`."""
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed int4 storage must be a uint8 matrix")
    if in_features < 0:
        raise ValueError("in_features must be non-negative")
    need = (in_features + 1) // 2
    if packed.shape[1] < need:
        raise ValueError("packed row is shorter than in_features")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty(
        packed.shape[0],
        packed.shape[1] * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :in_features].contiguous()
