# Inspect CUDA memory

Reports physical device memory when the selected backend exposes it and
allocator-owned current and peak bytes when those counters are
available. The native backend reports physical CUDA-driver memory plus
allocations owned by cudaverse. The optional torch backend reports its
allocator's allocated and reserved bytes. CPU selection returns an
unavailable report rather than pretending host RAM is CUDA memory.

## Usage

``` r
cuda_memory_info(device = c("auto", "cuda", "cpu"))
```

## Arguments

- device:

  Requested device: `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A `cuda_memory_info` list with selection metadata, physical
`total_bytes`, `free_bytes`, and `used_bytes`, allocator
`allocated_bytes`, `allocated_peak_bytes`, `reserved_bytes`, and
`reserved_peak_bytes`, plus `reason` and any captured `error`.
Unsupported counters are `NA_real_` rather than estimated.

## Details

This function does not reset allocator peaks or retain a user tensor.
The first CUDA selection in an R session can run the small runtime
self-test, so native peak bytes can include its released temporary
allocations. An automatic request is safe on a machine without CUDA and
records the CPU fallback. An explicit `device = "cuda"` request remains
strict.

## Examples

``` r
cuda_memory_info("cpu")
#> <cuda_memory_info available=FALSE device=cpu backend=base reason=cpu_backend_selected>
cuda_memory_info("auto")
#> <cuda_memory_info available=FALSE device=cpu backend=base reason=cpu_backend_selected>
```
