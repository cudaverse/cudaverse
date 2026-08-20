# Create a GPU-aware sparse matrix

Create a GPU-aware sparse matrix

## Usage

``` r
cuda_sparse(
  x,
  format = c("csr", "coo"),
  device = c("auto", "cuda", "cpu"),
  drop_zeros = TRUE
)
```

## Arguments

- x:

  A numeric matrix, a sparse matrix from the `Matrix` package, or a
  `cudasparse` object.

- format:

  Logical storage format, `"csr"` or `"coo"`.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

- drop_zeros:

  Whether to remove explicitly stored zeros.

## Value

A `cudasparse` list. Stable public metadata include one-based COO `i`
and `j`, numeric `values`, zero-based CSR `row_ptr` and `col_index`,
integer `shape`, matrix `dimnames`, logical `format`, actual `device`,
and `backend`. `storage` is backend-internal and should not be accessed
directly.

## Details

Existing `cudasparse` inputs use their stable sorted COO mirror
directly. Same-device format changes share backend storage; transfers
and zero filtering do not construct an intermediate Matrix object.

## Examples

``` r
library(Matrix)
x <- rsparsematrix(5, 4, density = 0.25)
cuda_sparse(x, device = "cpu")
#> <cudasparse[5x4] nnz=5 format=csr device=cpu backend=Matrix>
#> 5 x 4 sparse Matrix of class "dgCMatrix"
#>                        
#> [1,]  .    .     .    .
#> [2,]  .    .    -0.97 .
#> [3,]  .    0.69  .    .
#> [4,] -0.96 0.80  .    .
#> [5,] -0.48 .     .    .
```
