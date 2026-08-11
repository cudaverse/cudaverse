# Transpose a GPU-aware sparse matrix

Swaps sparse rows and columns while preserving stored values, logical
format, dimension labels, and the actual device. The native CUDA backend
transposes its CSR backing storage on the device. Compatibility backends
rebuild same-device storage from the stable public COO metadata.

## Usage

``` r
# S3 method for class 'cudasparse'
t(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

A transposed `cudasparse` matrix on the same device as `x`.

## Examples

``` r
x <- cuda_sparse(matrix(c(1, 0, 2, 0, 3, 0), 2), device = "cpu")
t(x)
#> <cudasparse[3x2] nnz=3 format=csr device=cpu backend=Matrix>
#> 3 x 2 sparse Matrix of class "dgCMatrix"
#>         
#> [1,] 1 .
#> [2,] 2 .
#> [3,] 3 .
```
