# t-SNE embedding

t-SNE currently uses the CPU `Rtsne` backend.

## Usage

``` r
cuda_tsne(
  x,
  n_components = 2L,
  perplexity = 30,
  theta = 0.5,
  seed = NULL,
  ...,
  reduced_dim = NULL
)
```

## Arguments

- x:

  Numeric observation-by-feature matrix, compatible cudaverse result, or
  a `SingleCellExperiment` with a reduced dimension.

- n_components:

  Output dimensions.

- perplexity:

  t-SNE perplexity.

- theta:

  Barnes-Hut accuracy/speed trade-off.

- seed:

  Optional random seed.

- ...:

  Additional arguments passed to
  [`Rtsne::Rtsne()`](https://rdrr.io/pkg/Rtsne/man/Rtsne.html).

- reduced_dim:

  For a `SingleCellExperiment`, the reduced-dimension name to embed.
  When `NULL`, a compatible recorded metadata choice is used first,
  followed by a uniquely named `"PCA"`. Other names must be selected
  explicitly.

## Value

A `cuda_embedding`; see
[`cuda_umap()`](https://cudaverse.github.io/cudaverse/reference/cuda_umap.md)
for the stable result fields.

## Examples

``` r
if (requireNamespace("Rtsne", quietly = TRUE)) {
  cuda_tsne(matrix(rnorm(120), 40, 3), perplexity = 5, seed = 1)
}
#> <cuda_embedding method=tsne observations=40 dimensions=2 backend=Rtsne compute_device=cpu>
```
