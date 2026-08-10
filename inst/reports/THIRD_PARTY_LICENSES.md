# Third-party software and redistribution status

This source package and its current CI artifacts do **not** bundle NVIDIA
libraries, LibTorch, or other third-party binary payloads.

| Component | How it is used | Bundled | License action |
|---|---|---:|---|
| NVIDIA CUDA Driver | Dynamically loaded from the user's system | No | Supplied by the NVIDIA driver installation |
| NVIDIA cuBLAS 12 | Dynamically loaded from a user-provided/system path | No | Do not redistribute until the exact files are checked against the current CUDA EULA Attachment A |
| NVIDIA cuSOLVER 11 | Dynamically loaded from a user-provided/system path | No | Do not redistribute until the exact files are checked against the current CUDA EULA Attachment A |
| NVIDIA CUDA Compiler 12.8.1 | Builds PTX in a pinned CI container only | No | Build tool; no compiler or Toolkit library is copied into the package |
| `cudaverse_dense_kernels.ptx` | Forward-compatible device code compiled from this project's MIT-licensed source | Yes | Package-owned build output; source, pinned build command, and SHA-256 are retained and reproducibility-tested |
| R | C API and package build system | No | Runtime/build dependency |

The release gate fails if a CUDA, cuBLAS, cuSPARSE, cuSOLVER, cuDNN, NVRTC,
driver, cubin/fatbin, or LibTorch binary appears in a built package. The only
allowed PTX is the checksum-pinned file built from this repository's own CUDA
source. A future binary distribution may only include exact runtime files
permitted by the current
[NVIDIA CUDA EULA Attachment A](https://docs.nvidia.com/cuda/eula/) and must
carry all notices required there.
