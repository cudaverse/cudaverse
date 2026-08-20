# GPU-aware singular value decomposition

GPU-aware singular value decomposition

## Usage

``` r
cuda_svd(
  x,
  nu = min(nrow(x), ncol(x)),
  nv = min(nrow(x), ncol(x)),
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x:

  A finite numeric matrix or `cudatensor`.

- nu, nv:

  Number of left and right singular vectors to return.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A list with `d`, `u`, `v`, and the actual `device`. Matrix row and
column names are retained on the corresponding singular vectors.

## Details

A native CUDA `cudatensor` selected on the same backend is validated for
finite values on the device and passed directly to cuSOLVER. Float32 and
integer storage is converted to float64 on the device. Only the
requested decomposition results are materialized in R.

## Examples

``` r
cuda_svd(matrix(rnorm(30), 10, 3), device = "cpu")
#> <cuda_svd rank=3 device=cpu compute=cpu backend=base>
```
