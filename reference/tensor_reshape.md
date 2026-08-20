# Reshape a tensor without changing its values

Reshape a tensor without changing its values

## Usage

``` r
tensor_reshape(x, shape)
```

## Arguments

- x:

  A `cudatensor`.

- shape:

  Positive whole-number dimensions whose product equals `length(x)`.

## Value

A `cudatensor` on the same device with the requested shape.

## Details

The native CUDA backend creates an allocation-free metadata view that
shares the source device allocation. The source and reshaped tensor have
independent external-pointer lifetimes, and the allocation is freed only
after the final view is released. Compatibility backends retain their
established reshape behavior.

## Examples

``` r
x <- cuda_tensor(1:6, device = "cpu")
tensor_reshape(x, c(2, 3))
#> <cudatensor[2x3] device=cpu backend=base dtype=integer>
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
```
