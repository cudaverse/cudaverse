# Create a GPU-aware tensor

Create a GPU-aware tensor

## Usage

``` r
cuda_tensor(x, device = c("auto", "cuda", "cpu"), dtype = NULL)
```

## Arguments

- x:

  Numeric vector, matrix, array, or another `cudatensor`.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`. Auto selects CUDA only when
  [`cuda_available()`](https://cudaverse.github.io/cudaverse/reference/cuda_available.md)
  is true.

- dtype:

  One of `"float64"`, `"float32"`, or `"integer"`.

  Matrix and array dimnames, including names on a one-dimensional input,
  are retained as R metadata on both CPU and CUDA tensors. Floating
  dtypes accept IEEE `Inf`, `-Inf`, `NaN`, and R's floating `NA`; torch
  backends may normalize `NA` to `NaN`. Integer dtype rejects non-finite
  or fractional values because they have no exact integer
  representation. When `x` is already a tensor on the selected device,
  compatible floating dtype changes use its current backend without
  materializing the tensor on the host. Conversion to integer still
  validates exact representability on the host.

## Value

A `cudatensor` object.

## Examples

``` r
x <- cuda_tensor(matrix(1:6, nrow = 2), device = "cpu")
x
#> <cudatensor[2x3] device=cpu backend=base dtype=integer>
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
```
