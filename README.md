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

CUDA is optional. The development line can discover the lightweight
[`cudaverseCUDA`](https://github.com/cudaverse/cudaverseCUDA) extension or use
a supported CUDA-enabled `torch` installation. When neither is available,
functions use their documented portable backend and record what actually ran.

## Lightweight native CUDA direction

Version 0.1.0 uses `torch` only as an optional CUDA backend; it is not a hard
package dependency. Development version 0.2 now has a backend registry and a
separate `cudaverseCUDA` extension behind the same public R API. Dense Phase 2
implements driver detection, shared-ownership device allocations, transfer,
casts, reductions, cuBLAS matrix multiplication, cuSOLVER SVD/PCA, exact
distance blocks, and deterministic top-k/kNN. Its measured design benefits
are:

- avoid requiring the full LibTorch installation, which occupied 6.86 GB in
  our Windows RTX 2000 development environment;
- remove coupling to changes in torch's R indexing, reshape semantics, and
  release cycle;
- keep PCA, distance calculation, and top-k selection resident on the GPU
  instead of repeatedly transferring intermediate results;
- control `cudatensor` and `cudasparse` memory layout, lifetime, and compute
  provenance directly; and
- allow future backends to be added without changing user code.

These are development-line results, not claims about release 0.1.0. Native
CUDA remains opt-in in 0.2 because element-wise/broadcast and sparse CUDA paths
do not yet satisfy the same full-surface gate; making it the global automatic
choice would regress currently supported torch workflows. CPU and torch
compatibility behavior is unchanged, and sparse CUDA remains the next
milestone. Reproducible PTX, RTX 2000 parity/lifecycle evidence, the CycloneDX
SBOM, and the third-party redistribution inventory are published with the
extension source. See the
[native CUDA roadmap](.github/NATIVE-CUDA-ROADMAP.md) for the architecture and
acceptance criteria.

## Installation

During development, install from GitHub:

```r
# install.packages("pak")
pak::pak("cudaverse/cudaverse")
```

Native CUDA development uses the optional extension. It is discovered lazily,
so installing `cudaverse` never downloads a CUDA runtime or requires a CUDA
toolkit:

```r
pak::pak("cudaverse/cudaverse@develop/native-cuda")
pak::pak("cudaverse/cudaverseCUDA")

# During Phase 2, request native before the compatibility backend.
options(cudaverse.cuda_backends = c("native", "torch"))
cuda_diagnostics()
```

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

Single-cell-specific workflows live in the separate `cudacellr` extension so
general users do not need the SingleCellExperiment or Seurat ecosystems.

## Project history

The original component repositories remain available as archived development
history. New features, bug reports, documentation, and releases for the
general-purpose API belong in this repository.
