"""Post-training integer quantization of linear weights."""

from ptqkit.convert import (
    compressed_state,
    iter_targets,
    matrix_nbytes,
    payload_nbytes,
    quantize_model,
    save_compressed,
    state_dict_nbytes,
    weight_out_in,
)
from ptqkit.gemm import dequant_gemm, kernel_info
from ptqkit.linear import QuantizedLinear
from ptqkit.pack import pack_int4, unpack_int4
from ptqkit.quant import (
    Method,
    QuantConfig,
    QuantizedWeight,
    absolute_error_bound,
    dequantize_weight,
    quantize_weight,
)

__version__ = "0.1.0"

__all__ = [
    "Method",
    "QuantConfig",
    "QuantizedLinear",
    "QuantizedWeight",
    "absolute_error_bound",
    "compressed_state",
    "dequant_gemm",
    "dequantize_weight",
    "iter_targets",
    "kernel_info",
    "matrix_nbytes",
    "pack_int4",
    "payload_nbytes",
    "quantize_model",
    "quantize_weight",
    "save_compressed",
    "state_dict_nbytes",
    "unpack_int4",
    "weight_out_in",
]
