# GPU-aware principal component analysis

GPU-aware principal component analysis

## Usage

``` r
cuda_pca(
  x,
  n_components = 2L,
  center = TRUE,
  scale. = FALSE,
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x:

  A matrix or `cudasparse` object with observations in rows and features
  in columns.

- n_components:

  Number of components to return.

- center:

  Whether to centre features.

- scale.:

  Whether to scale features to unit variance.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A `cuda_pca` object with scores in `x`, loadings in `rotation`, standard
deviations, centring/scaling values, and actual device. Observation
names, feature names, and stable `PC1`, `PC2`, ... component names are
preserved on every backend.

## Examples

``` r
fit <- cuda_pca(iris[, 1:4], n_components = 2, device = "cpu")
fit
#> <cuda_pca components=2 device=cpu compute=cpu backend=stats>
#>                      PC1         PC2
#> Sepal.Length  0.36138659 -0.65658877
#> Sepal.Width  -0.08452251 -0.73016143
#> Petal.Length  0.85667061  0.17337266
#> Petal.Width   0.35828920  0.07548102
```
