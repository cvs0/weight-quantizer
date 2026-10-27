"""Build the int8/int4 dequant-matmul extension."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _cpu_flag(name: str) -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            tokens = handle.read().split()
    except OSError:
        return False
    return name in tokens


extra_compile = ["-O3", "-fopenmp"]
if _cpu_flag("avx2"):
    extra_compile.extend(["-mavx2", "-mfma"])

ext_modules = [
    Pybind11Extension(
        "ptqkit._kernels",
        ["src/ptqkit/csrc/dequant_gemm.cpp"],
        extra_compile_args=extra_compile,
        extra_link_args=["-fopenmp"],
        cxx_std=17,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
