from typing import Any

def dequant_gemm(
    x: Any,
    q: Any,
    scales: Any,
    zeros: Any,
    channel_scale: Any,
    group_size: int,
    bits: int,
    nthreads: int,
) -> Any: ...
def kernel_info() -> dict[str, bool]: ...
