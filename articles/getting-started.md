# Getting started with cudaverse

`cudaverse` is a single entry point for dense tensors, sparse matrices,
numerical algorithms, graphs, and embeddings. CUDA is optional: every
result records the backend and device that actually performed each
stage.

## Dense and sparse data

``` r

library(cudaverse)

dense <- cuda_tensor(matrix(1:12, nrow = 4), device = "cpu")
tensor_shape(dense)
#> [1] 4 3

sparse <- cuda_sparse(Matrix::Diagonal(4), device = "cpu")
sparse_info(sparse)
#> $shape
#> [1] 4 4
#> 
#> $nnz
#> [1] 4
#> 
#> $density
#> [1] 0.25
#> 
#> $format
#> [1] "csr"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $provenance_schema
#> [1] "cudaverse-stage/1"
#> 
#> $compute_device
#> [1] "cpu"
```

Sparse normalization keeps the zero pattern and composes with the same
PCA and kNN entry points:

``` r

counts <- matrix(c(1, 0, 3, 2, 4, 1, 0, 2, 5, 1, 3, 2), nrow = 4)
sparse_counts <- cuda_sparse(counts, device = "cpu")
normalized <- sparse_normalize(
  sparse_counts, margin = "rows", scale_factor = 100, log1p = TRUE
)
sparse_pca <- cuda_pca(normalized, n_components = 2, device = "cpu")
sparse_neighbors <- cuda_knn(sparse_pca$x, k = 2, device = "cpu")
```

## Algorithms compose directly

``` r

set.seed(1)
x <- matrix(rnorm(120), nrow = 30)

pca <- cuda_pca(x, n_components = 4, device = "cpu")
neighbors <- cuda_knn(pca$x, k = 4, device = "cpu")
graph <- cuda_knn_graph(neighbors)

dim(pca$x)
#> [1] 30  4
dim(as_adjacency_matrix(graph))
#> [1] 30 30
cuda_provenance(pca)
#> <cuda_provenance schema=cudaverse-stage/1 stages=2 compute=cpu>
#>          stage requested_device device backend selection_reason fallback
#>  preprocessing              cpu    cpu   stats     explicit_cpu    FALSE
#>  decomposition              cpu    cpu   stats     explicit_cpu    FALSE
#>  output_device
#>            cpu
#>            cpu
```

UMAP, t-SNE, Louvain, and Leiden use optional packages. Their functions
report a clear installation message when the relevant optional backend
is missing.

## Single-cell extension

Single-cell normalization, feature selection, and native
SingleCellExperiment/Seurat mapping remain in `cudacellr`. That package
depends on `cudaverse`, so a single-cell user installs two packages
while a general numerical user installs only this one.
