# GPU-aware k-means clustering

GPU-aware k-means clustering

## Usage

``` r
cuda_kmeans(
  x,
  centers,
  iter.max = 100L,
  tolerance = 1e-06,
  seed = NULL,
  batch_size = 256L,
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x:

  Numeric matrix with observations in rows.

- centers:

  Number of clusters or a matrix of initial centres.

- iter.max:

  Maximum Lloyd iterations.

- tolerance:

  Convergence tolerance for centre movement.

- seed:

  Optional random seed used for initial centres.

- batch_size:

  Maximum number of observations whose centre-distance block is
  materialized at once. The native backend keeps observations, centres,
  assignments, and updates on the GPU while bounding temporary distance
  storage to approximately `batch_size * n_centers` values.

- device:

  Device used for the numerical clustering stages.

## Value

A `cuda_kmeans` list containing integer `cluster` assignments, final
`centers`, per-cluster `withinss`, `tot.withinss`, the number of
iteration count in `iter`, a logical `converged` flag, and the actual
distance `device`. Observation and feature names are retained when
supplied.

## Details

The native CUDA backend uploads the observations and initial centres
once, then keeps distance calculation, deterministic assignment,
accumulation, and centre updates on the device. Only the small
convergence movement summary is inspected between iterations; final
assignments, centres, and within-cluster sums are transferred to R.
Compatibility backends without a resident k-means operation retain the
established distance-on-backend and update-on-CPU implementation.

## Examples

``` r
set.seed(1)
x <- rbind(matrix(rnorm(40), 20, 2), matrix(rnorm(40, 4), 20, 2))
cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
#> <cuda_kmeans clusters=2 iterations=2 converged=TRUE distance_device=cpu compute=cpu backend=base>
#>           [,1]         [,2]
#> [1,] 0.1905239 -0.006471519
#> [2,] 4.1387968  4.101736906
```
