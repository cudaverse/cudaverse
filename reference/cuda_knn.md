# k-nearest neighbours

k-nearest neighbours

## Usage

``` r
cuda_knn(
  x,
  k = 15L,
  metric = c("euclidean", "cosine"),
  device = c("auto", "cuda", "cpu"),
  batch_size = 256L
)
```

## Arguments

- x:

  Numeric matrix or `cudasparse` object with observations in rows.

- k:

  Number of neighbours.

- metric:

  Exact distance metric, `"euclidean"` or `"cosine"`.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

- batch_size:

  Maximum number of query rows in each dense distance block. Larger
  batches may be faster but use more memory.

## Value

A `cuda_knn` list with `index` and `distance` matrices of size `nrow(x)`
by `k`, followed by the selected `metric` and actual `device`.
Neighbours in every row are ordered by distance and then row index. When
`x` has row names, both matrices retain them as query identifiers;
neighbour identities can be recovered with
`rownames(result$index)[result$index]`.

## Details

Neighbours are exact: every row is compared with every other row. The
observation itself is always excluded. Equal distances are resolved
deterministically in favour of the smaller row index.

The implementation constructs at most a
`min(batch_size, nrow(x))`-by-`nrow(x)` dense distance block instead of
a complete pairwise distance matrix. The native CUDA backend keeps
distance blocks and deterministic top-k selection on the GPU, then
transfers only the final `n`-by-`k` index and distance matrices.
Compatibility backends without device-side selection transfer each
distance block to the CPU for stable ordering. On CPU, Euclidean blocks
use the same guarded translated-and-scaled implementation as
[`cuda_distance()`](https://cudaverse.github.io/cudaverse/reference/cuda_distance.md).

## Examples

``` r
cuda_knn(
  matrix(rnorm(30), 10, 3),
  k = 3,
  batch_size = 4,
  device = "cpu"
)
#> <cuda_knn observations=10 k=3 metric=euclidean distance_device=cpu compute=cpu backend=base>
```
