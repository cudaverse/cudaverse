# Normalize sparse rows or columns without densifying

Each selected row or column is divided by its sum and multiplied by
`scale_factor`. Optionally, [`log1p()`](https://rdrr.io/r/base/Log.html)
is applied to stored non-zero values. The operation preserves sparse
structure and dimension labels.

## Usage

``` r
sparse_normalize(
  x,
  margin = c("rows", "columns"),
  scale_factor = 1,
  log1p = FALSE
)
```

## Arguments

- x:

  A non-negative `cudasparse` matrix.

- margin:

  Normalize `"rows"` or `"columns"`.

- scale_factor:

  Positive target sum before the optional log transform.

- log1p:

  Whether to apply [`log1p()`](https://rdrr.io/r/base/Log.html) to
  normalized stored values.

## Value

A `cudasparse` matrix on the same device as `x`.

## Examples

``` r
x <- cuda_sparse(matrix(c(1, 0, 3, 2), 2), device = "cpu")
sparse_normalize(x, margin = "rows", scale_factor = 1)
#> <cudasparse[2x2] nnz=3 format=csr device=cpu backend=Matrix>
#> 2 x 2 sparse Matrix of class "dgCMatrix"
#>               
#> [1,] 0.25 0.75
#> [2,] .    1.00
```
