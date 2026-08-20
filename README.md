# cudaverse

**Lightweight CUDA computing for R.** `cudaverse` lets R users run dense and
sparse matrix operations, PCA, distances, exact k-nearest neighbours, and
clustering on an NVIDIA GPU through one familiar package.

Use cudaverse when your analysis has outgrown ordinary R execution and you
want GPU acceleration without adopting a large deep-learning framework. The
native backend is included in the package, discovers CUDA libraries already
installed on your system, and keeps supported multi-stage workflows on the GPU
until you ask for an R result.

## Why use cudaverse?

- **One R interface:** start with a matrix or a `Matrix` sparse matrix and use
  ordinary R functions such as `t()`, `[`, and `%*%` alongside `cuda_pca()` and
  `cuda_knn()`.
- **Lightweight:** cudaverse does not bundle LibTorch or a CUDA runtime. In the
  retained 0.4 benchmark, the installed package was about 1.45 MB; the optional
  R `torch` installation used for comparison occupied about 7.37 GB.
- **GPU-resident pipelines:** native PCA, distance blocks, stable top-k, exact
  kNN, k-means, and supported sparse continuations reuse device memory instead
  of repeatedly copying intermediates to R.
- **Strict CUDA execution:** the examples use `device = "cuda"`. If CUDA is not
  ready, cudaverse reports what is missing instead of silently running the
  requested GPU task elsewhere.
- **Auditable results:** `cuda_provenance()` records the backend, device, and
  host/device boundary for every computational stage.

## Requirements

GPU execution requires a CUDA-capable NVIDIA GPU and compatible NVIDIA runtime
libraries. The easiest supported setup is:

- **Windows:** install a current NVIDIA driver and a compatible CUDA 12.x
  distribution that provides `cublas64_12.dll` and `cusolver64_11.dll`.
- **Linux:** install a current NVIDIA driver plus CUDA 12.x runtime libraries
  that provide `libcuda.so`, `libcublas.so.12`, and `libcusolver.so.11`.
- **macOS:** current NVIDIA CUDA does not support macOS. The R package can be
  installed and checked there, but CUDA execution requires Windows or Linux.

Use NVIDIA's current
[Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
or [Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
After installing the driver, confirm that `nvidia-smi` can see the GPU before
starting R.

## Install

Install the current release from GitHub:

```r
# install.packages("pak")
pak::pak("cudaverse/cudaverse@v0.4.1")
```

Then verify the complete runtime—not only GPU detection:

```r
library(cudaverse)

diagnostics <- cuda_diagnostics()
diagnostics$summary
diagnostics$next_steps

# Strict: returns a CUDA selection or explains why CUDA is not ready.
cuda_select_device("cuda")
```

The native backend needs the NVIDIA driver, cuBLAS 12, and cuSOLVER 11. If
Windows or Linux cannot find the libraries automatically, set
`CUDAVERSE_CUBLAS_PATH` and `CUDAVERSE_CUSOLVER_PATH` to their absolute paths
before loading cudaverse. See the
[CUDA setup guide](https://cudaverse.github.io/cudaverse/articles/gpu-setup.html)
for platform-specific examples and troubleshooting.

## A five-minute CUDA workflow

The following workflow uploads a matrix once, computes PCA and exact kNN with
CUDA, and transfers only the final neighbour result back to R:

```r
library(cudaverse)

set.seed(1)
x <- matrix(rnorm(10000 * 100), nrow = 10000, ncol = 100)

pca <- cuda_pca(
  x,
  n_components = 20,
  device = "cuda"
)

neighbors <- cuda_knn(
  pca$x,
  k = 15,
  device = "cuda"
)

head(neighbors$index)
head(neighbors$distance)
cuda_provenance(neighbors)
```

Use tensors when you want direct control over GPU-resident matrix operations:

```r
x_gpu <- cuda_tensor(x, device = "cuda", dtype = "float32")
crossprod_gpu <- tensor_matmul(t(x_gpu), x_gpu)

tensor_device(crossprod_gpu)
crossprod <- to_cpu(crossprod_gpu)  # transfer only when the R matrix is needed
```

Sparse matrices use the same CUDA-facing API:

```r
counts <- Matrix::rsparsematrix(10000, 100, density = 0.03)
counts@x <- abs(counts@x)

counts_gpu <- cuda_sparse(counts, device = "cuda")
normalized_gpu <- sparse_normalize(
  counts_gpu,
  margin = "rows",
  scale_factor = 10000,
  log1p = TRUE
)
sparse_pca <- cuda_pca(normalized_gpu, n_components = 20, device = "cuda")
sparse_knn <- cuda_knn(sparse_pca$x, k = 15, device = "cuda")
```

## How fast is it?

The retained full benchmark uses five warmups and ten timed runs and validates
numerical parity before accepting a timing. Selected host-boundary medians are
shown below; times include the public R call boundary.

| Workload | Base R | Native CUDA | R `torch` | Native vs base | Native vs `torch` |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024 float32 matrix multiplication | 0.332 s | 0.024 s | 0.039 s | 13.7x | 1.6x |
| Dense PCA + exact kNN, 10,000 x 100 | 11.915 s | 0.275 s | 6.094 s | 43.4x | 22.2x |
| Sparse PCA + exact kNN, 10,000 x 100 | 10.265 s | 0.203 s | 5.198 s | 50.6x | 25.6x |
| Dense PCA + exact kNN, 50,000 x 128 | 668.418 s | 2.807 s | 126.552 s | 238.1x | 45.1x |

These are workload-specific measurements from one fixed NVIDIA GPU
environment, not guarantees for every machine. Small jobs can be dominated by
launch and transfer overhead: in the retained sparse 1,000 x 50 case, base R
was faster than native CUDA. See the complete
[benchmark evidence](https://github.com/cudaverse/cudaverse/blob/main/inst/reports/benchmarks/CP07-FULL.md) and the
[performance articles](https://cudaverse.github.io/cudaverse/articles/)
before choosing a workload size.

The versioned
[benchmark contract](https://github.com/cudaverse/cudaverse/blob/0a422fad7744da4116915037fc7134075a65e2a0/inst/benchmarks/README.md)
defines the workload sizes, warmups, timed runs, transfer boundaries,
validation tolerances, and evidence fields used by the retained report.

## Learn by task

- [First CUDA workflow](https://cudaverse.github.io/cudaverse/articles/getting-started.html)
- [Windows and Linux CUDA setup](https://cudaverse.github.io/cudaverse/articles/gpu-setup.html)
- [Keep intermediates on the GPU](https://cudaverse.github.io/cudaverse/articles/backend-provenance.html)
- [Supported CUDA operations](https://cudaverse.github.io/cudaverse/articles/backend-support.html)
- [Benchmarks and comparison](https://cudaverse.github.io/cudaverse/articles/)

Graph and embedding helpers are also available. Some currently delegate stages
to established CPU packages; `cuda_provenance()` makes those boundaries
visible. The task guides identify which operations are fully native CUDA and
which are hybrid.
