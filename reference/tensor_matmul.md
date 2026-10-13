# Matrix multiplication for tensors

Matrix multiplication for tensors

## Usage

``` r
tensor_matmul(x, y, precision = "standard")
```

## Arguments

- x, y:

  Two-dimensional `cudatensor` objects or numeric matrices.

- precision:

  Compute policy: `"standard"` preserves the existing dtype and backend
  arithmetic. `"tf32"` explicitly permits reduced-precision
  TensorFloat-32 multiplication with float32 output on the native CUDA
  backend. It requires two native CUDA float32 tensors and an NVIDIA GPU
  with compute capability 8.0 or newer.

## Value

A `cudatensor`.

## Details

Row names come from `x` and column names come from `y`. When both
operands name the contracted dimension, those names must be identical.

Standard native float32 multiplication uses cuBLAS SGEMM; float64 uses
DGEMM. Float32 inputs are never automatically promoted to float64 for
accuracy. TF32 permits cuBLAS to use Tensor Cores with reduced input
precision; the selected kernel and any speedup depend on the workload.
An unsupported TF32 request raises an error, including when
`NVIDIA_TF32_OVERRIDE=0`. The policy applies only to this call, and
[`cuda_provenance()`](https://cudaverse.github.io/cudaverse/reference/cuda_provenance.md)
records `matmul_standard` or `matmul_tf32_permitted` as its selection
reason. `%*%` uses the standard policy.

Rounding and cancellation can cause float32 dot products to differ from
a double-precision reference, especially near zero. TF32 allows larger
differences. Use float64 tensors when the analysis requires greater
accuracy; assess precision against the needs of the complete workflow.

## Examples

``` r
x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")
tensor_matmul(x, y)
#> <cudatensor[2x2] device=cpu backend=base dtype=float64>
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64
```
