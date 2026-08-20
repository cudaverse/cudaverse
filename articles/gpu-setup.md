# Set up CUDA for cudaverse

Follow this guide once on the Windows or Linux computer that will run
cudaverse. At the end, a short R example verifies the complete setup.

## Platform support

| Platform | CUDA with cudaverse | Required runtime |
|----|----|----|
| Windows 10/11 or Windows Server | Supported | NVIDIA driver, CUDA Driver API, cuBLAS 12, cuSOLVER 11 |
| Supported Linux distribution | Supported | NVIDIA driver, `libcuda`, cuBLAS 12, cuSOLVER 11 |
| macOS | Not supported by current NVIDIA CUDA | Use a Windows or Linux machine for CUDA work |

NVIDIA changes its supported operating-system and driver matrix over
time. Use the current official [Windows CUDA
guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
or [Linux CUDA
guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
rather than copying an old installation command.

## 1. Install and verify the NVIDIA driver

Install a current driver for the CUDA-capable NVIDIA GPU. Open
PowerShell, Command Prompt, or a Linux shell and run:

``` text
nvidia-smi
```

Do not continue until this command lists the GPU without a driver error.
The CUDA version shown by `nvidia-smi` describes driver compatibility;
it does not prove that cuBLAS and cuSOLVER are installed.

## 2. Install CUDA 12.x

cudaverse is lightweight because it uses the CUDA installation already
on your computer. It needs these NVIDIA components:

- the CUDA Driver API from the NVIDIA driver;
- cuBLAS major version 12; and
- cuSOLVER major version 11.

Installing a compatible CUDA 12.x distribution from NVIDIA is the
simplest way to obtain them. Use NVIDIA’s default installation choices
unless your system administrator manages CUDA centrally.

### Windows

Confirm that the CUDA `bin` directory contains:

``` text
cublas64_12.dll
cusolver64_11.dll
```

The NVIDIA installer normally adds its `bin` directory to `PATH`. If R
still cannot find the files, set their absolute paths before loading
cudaverse:

``` r

Sys.setenv(
  CUDAVERSE_CUBLAS_PATH =
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/bin/cublas64_12.dll",
  CUDAVERSE_CUSOLVER_PATH =
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/bin/cusolver64_11.dll"
)
```

Replace `v12.x` with the installed CUDA directory. Use forward slashes
or escaped backslashes in R paths.

### Linux

Ask the dynamic loader whether the libraries are visible:

``` text
ldconfig -p | grep -E 'libcuda.so|libcublas.so.12|libcusolver.so.11'
```

If they are installed outside the loader’s configured paths, either
configure the system loader according to the NVIDIA guide or set
absolute paths before loading cudaverse:

``` r

Sys.setenv(
  CUDAVERSE_CUBLAS_PATH = "/absolute/path/to/libcublas.so.12",
  CUDAVERSE_CUSOLVER_PATH = "/absolute/path/to/libcusolver.so.11"
)
```

Avoid pointing at an unversioned symlink from an incompatible CUDA
release. The runtime self-test, not the directory name, is the final
compatibility check.

### macOS

CUDA 10.2 was NVIDIA’s final CUDA release for macOS. Current cudaverse
native CUDA targets Windows and Linux and reports CUDA as unavailable on
macOS. Use a Windows/Linux workstation, server, or cloud instance with
an NVIDIA GPU for the workflows in these guides.

## 3. Install cudaverse

``` r

# install.packages("pak")
pak::pak("cudaverse/cudaverse@v0.4.1")
library(cudaverse)
```

The installed R package remains small because it does not download
LibTorch or copy NVIDIA runtime libraries into the package library.

## 4. Check that cudaverse can use CUDA

Run diagnostics in a fresh R session after changing a driver, `PATH`,
loader configuration, or either cudaverse library-path variable:

``` r

check <- cuda_diagnostics()
check$summary
check$next_steps
```

The summary checks the GPU, driver, CUDA libraries, and a small
calculation. If anything is missing, `next_steps` tells you what to fix.

## 5. Run a strict CUDA smoke test

This test requires CUDA and does not silently change devices:

``` r

cuda_select_device("cuda")

set.seed(1)
x_gpu <- cuda_tensor(
  matrix(rnorm(1024^2), nrow = 1024),
  device = "cuda",
  dtype = "float32"
)
y_gpu <- tensor_matmul(x_gpu, x_gpu)

tensor_device(y_gpu)
cuda_provenance(y_gpu)
cuda_memory_info("cuda")
```

For a successful lightweight run, provenance reports `device = "cuda"`
and `backend = "native"` for the matrix multiplication.

## Common problems

### `nvidia-smi` fails

Repair or update the NVIDIA driver first. cudaverse cannot load the CUDA
Driver API when the operating system cannot communicate with the GPU.

### `cublas_loaded` or `cusolver_loaded` is false

Install the compatible CUDA 12.x runtime libraries or set the two
absolute library paths before loading cudaverse. Restart R afterward.

### CUDA is found, but the test calculation fails

Read `check$next_steps`. Restart R after changing the driver or CUDA
installation, then run the diagnostic check again.

### CUDA runs out of memory

- Reduce `batch_size` for
  [`cuda_distance()`](https://cudaverse.github.io/cudaverse/reference/cuda_distance.md)
  or
  [`cuda_knn()`](https://cudaverse.github.io/cudaverse/reference/cuda_knn.md).
- Prefer
  [`cuda_knn()`](https://cudaverse.github.io/cudaverse/reference/cuda_knn.md)
  over a complete pairwise distance matrix when only neighbours are
  needed.
- Remove unused GPU objects and run
  [`gc()`](https://rdrr.io/r/base/gc.html) before measuring a suspected
  leak.
- Inspect `cuda_memory_info("cuda")`; whole-device usage can include
  other applications.

### A graph or embedding result contains a non-CUDA stage

Some graph clustering and embedding functions currently delegate stages
to established R packages. This is documented rather than hidden. Use
`cuda_provenance(result)` and the [operation coverage
guide](https://cudaverse.github.io/cudaverse/articles/backend-support.md)
to distinguish native CUDA tasks from hybrid workflows.

R `torch` is not required for the lightweight native CUDA path.
