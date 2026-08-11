# Detect a usable CUDA backend

Detection uses the built-in lightweight native backend or a CUDA-enabled
installation of the optional `torch` package. NVIDIA libraries are
loaded only when diagnostics or CUDA selection is requested.

## Usage

``` r
cuda_available()
```

## Value

A single logical value.

## Examples

``` r
cuda_available()
#> [1] FALSE
```
