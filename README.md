# cudaverse

`cudaverse` gives R users one package for GPU-aware numerical workflows. It
combines the former `cudatensr`, `cudasparsr`, `cudalearnr`, `cudagraphR`, and
`cudaembedr` projects behind a single installation and documentation site.

The public API is organized by topic rather than by package:

- device selection and compute provenance;
- dense tensors and sparse matrices;
- SVD, PCA, distances, k-nearest neighbours, and k-means;
- weighted kNN graphs, Louvain, and Leiden clustering;
- UMAP, t-SNE, and diffusion-map-style embeddings.

CUDA is optional. The development line includes a lightweight native backend
and can also use a supported CUDA-enabled `torch` installation. NVIDIA
libraries are discovered only when CUDA diagnostics or selection is requested.
When neither CUDA path is available, functions use their documented portable
backend and record what actually ran.

## Lightweight native CUDA direction

Version 0.1.0 uses `torch` only as an optional CUDA backend; it is not a hard
package dependency. Development version 0.2 now has a backend registry and a
built-in native implementation behind the same public R API. Phase 3 added
shared-ownership COO/CSR storage, sparse multiplication and reductions,
sparse-preserving normalization, and sparse-input PCA/kNN to the validated
dense Phase 2 pipeline. Phase 4 adds the remaining arithmetic, broadcasting,
reshape, transpose, and float32 matmul surface needed for safe automatic
selection. Its measured design benefits are:

- avoid requiring the full LibTorch installation, which occupied 6.86 GB in
  our Windows RTX 2000 development environment;
- remove coupling to changes in torch's R indexing, reshape semantics, and
  release cycle;
- keep PCA, distance calculation, and top-k selection resident on the GPU
  instead of repeatedly transferring intermediate results;
- control `cudatensor` and `cudasparse` memory layout, lifetime, and compute
  provenance directly; and
- allow future backends to be added without changing user code.

These are development-line results, not claims about release 0.1.0. In the
Phase 4 candidate, native becomes the preferred CUDA backend for `device =
"auto"` only when its backend contract and capabilities match,
driver/cuBLAS/cuSOLVER/PTX components are healthy, and a
cached runtime self-test passes. An incomplete or unhealthy native runtime
cannot become automatic; torch remains the compatibility CUDA backend and CPU
remains the observable fallback. Explicit CUDA requests never silently fall
back. Reproducible PTX, RTX 2000
parity/lifecycle evidence, the CycloneDX SBOM, and the third-party
redistribution inventory are kept in this repository. See the
[native CUDA roadmap](.github/NATIVE-CUDA-ROADMAP.md) for the architecture and
  acceptance criteria, the historical
  [Phase 4 RTX report](inst/reports/native/STAGE4.md)
  for machine-backed evidence, and the
  [0.2 release-candidate assessment](.github/NATIVE-CUDA-PHASE4-RC.md) for the
  bounded release decision.

## Installation

During development, install from GitHub:

```r
# install.packages("pak")
pak::pak("cudaverse/cudaverse")
```

The native backend is included but remains runtime-lazy, so installing
`cudaverse` never downloads a CUDA runtime or requires a CUDA toolkit:

```r
pak::pak("cudaverse/cudaverse@develop/native-cuda")

diagnostics <- cuda_diagnostics()
diagnostics$selected_backend
diagnostics$auto_eligible_backends
diagnostics$backend_diagnostics$native$self_test
diagnostics$backend_diagnostics$native$capabilities
diagnostics$backend_diagnostics$native$operations
```

On Windows, `CUDAVERSE_CUBLAS_PATH` and `CUDAVERSE_CUSOLVER_PATH` may point
to user-provided `cublas64_12.dll` and `cusolver64_11.dll` files when those
libraries are not already on the loader path. The package does not copy or
redistribute them.

The option `cudaverse.cuda_backends` can still constrain backend order for
testing, but it cannot bypass native contract, runtime, or self-test gates.
The [backend support article](vignettes/backend-support.Rmd) distinguishes
direct, hybrid, CPU-only, metadata, probe, and host-materializing APIs across
the base, torch, and native backends.

## One workflow, one package

```r
library(cudaverse)

x <- matrix(rnorm(400), nrow = 40)
pca <- cuda_pca(x, n_components = 5)
neighbors <- cuda_knn(pca$x, k = 5)
graph <- cuda_knn_graph(neighbors)
embedding <- cuda_umap(pca$x)

cuda_provenance(pca)
embedding_coordinates(embedding)
```

Sparse inputs use the same algorithms and can remain device-resident through
normalization and PCA:

```r
counts <- Matrix::rsparsematrix(1000, 100, density = 0.05)
counts@x <- abs(counts@x)
sparse <- cuda_sparse(counts, device = "cuda")
normalized <- sparse_normalize(
  sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
)
pca <- cuda_pca(normalized, n_components = 20, device = "cuda")
neighbors <- cuda_knn(pca$x, k = 15, device = "cuda")
```

Single-cell-specific workflows live in the separate `cudacellr` extension so
general users do not need the SingleCellExperiment or Seurat ecosystems.

## Project history

The original component repositories remain available as archived development
history. New features, bug reports, documentation, and releases for the
general-purpose API belong in this repository.
