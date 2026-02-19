
from pathlib import Path
from setuptools import setup
import os
import sys

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


ROOT = Path(__file__).parent.resolve()

INCLUDE_DIRS = [
    str(ROOT / "backend/include"),
    str(ROOT / "thirdparty/eigen"),
]

CPU_SOURCES = [
    str(ROOT / "backend/src/gn.cpp"),
]

CUDA_SOURCES = [
    str(ROOT / "backend/src/gn_kernels.cu"),
    str(ROOT / "backend/src/matching_kernels.cu"),
]

# Base compile flags
extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-unused-variable",
        # Uncomment if you see long template instantiation times with Eigen
        # "-Wno-psabi",
    ],
}

# nvcc flags will be appended below if CUDA is available

def make_extension():
    # Require a CUDA toolkit for this project (kernels are mandatory)
    if not (torch.cuda.is_available() and CUDA_HOME):
        raise RuntimeError(
            "CUDA toolkit not found or CUDA not available. Make sure NVIDIA drivers are installed, "
            "nvcc is available, and you're building inside the same env as PyTorch."
        )

    sources = list(CPU_SOURCES) + list(CUDA_SOURCES)

    # Respect TORCH_CUDA_ARCH_LIST if provided; otherwise supply a safe default
    arch_env = (os.environ.get("TORCH_CUDA_ARCH_LIST") or "").strip()

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
    ]

    if not arch_env:
        # No env provided → embed common SASS + PTX for compute_90 (Hopper JIT).
        # NOTE: sm_90a isn't recognized by CUDA 12.1 toolchains. Using compute_90 PTX
        #       allows the driver to JIT for H100/H200 even without sm_90a SASS.
        nvcc_flags += [
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
            # PTX fallback for Hopper. Driver will JIT to the exact SM at runtime.
            "-gencode=arch=compute_90,code=compute_90",
        ]

    extra_compile_args["nvcc"] = nvcc_flags

    # Optional: define macros for Eigen or debug toggles
    define_macros = [
        ("EIGEN_MPL2_ONLY", None),
        # ("NDEBUG", None),  # uncomment for release-only builds
    ]

    ext = CUDAExtension(
        name="mast3r_slam_backends",
        sources=sources,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
    )
    return ext


ext_modules = [make_extension()]

setup(
    name="mast3r_slam_backends",
    version="0.1.0",
    description="MASt3R-SLAM CUDA backend",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

    