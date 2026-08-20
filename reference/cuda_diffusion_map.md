# Diffusion-map-style embedding

Pairwise distances can use the cudaverse CUDA path. Kernel construction
and eigendecomposition currently run on the CPU.

## Usage

``` r
cuda_diffusion_map(
  x,
  n_components = 2L,
  sigma = NULL,
  diffusion_time = 1,
  metric = c("euclidean", "cosine"),
  device = c("auto", "cuda", "cpu"),
  reduced_dim = NULL
)
```

## Arguments

- x:

  Numeric observation-by-feature matrix, compatible cudaverse result, or
  a `SingleCellExperiment` with a reduced dimension.

- n_components:

  Output dimensions.

- sigma:

  Gaussian kernel bandwidth. Defaults to the median positive pairwise
  distance.

- diffusion_time:

  Non-negative diffusion time exponent.

- metric:

  Euclidean or cosine distance.

- device:

  Device passed to
  [`cuda_distance()`](https://cudaverse.github.io/cudaverse/reference/cuda_distance.md).

- reduced_dim:

  For a `SingleCellExperiment`, the reduced-dimension name to embed. See
  [`cuda_umap()`](https://cudaverse.github.io/cudaverse/reference/cuda_umap.md)
  for automatic selection.

## Value

A `cuda_embedding` with the stable fields documented by
[`cuda_umap()`](https://cudaverse.github.io/cudaverse/reference/cuda_umap.md),
stage-level distance/kernel/eigendecomposition provenance, an optional
`distance_input` stage when resident native storage is reused, and an
additional `eigenvalues` element.

## Examples

``` r
cuda_diffusion_map(
  matrix(rnorm(120), 40, 3),
  n_components = 2,
  device = "cpu"
)
#> <cuda_embedding method=diffusion observations=40 dimensions=2 backend=base-eigen compute_device=cpu>
```
