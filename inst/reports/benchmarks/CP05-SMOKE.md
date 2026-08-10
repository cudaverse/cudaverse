# CP-05 benchmark-contract smoke evidence

The `cudaverse-benchmark/1` smoke profile completed from clean source commit
`cf40e4793ed33fe98bf7967641194aede0431d40` on an NVIDIA RTX 2000 Ada
Generation GPU (16,380 MiB, compute capability 8.9) with driver 595.97.
`tools/check-benchmark-report.R` accepted the generated report. Every base,
native, and torch case passed numerical and provenance validation.

This is a runner/contract smoke test, not a performance assessment. Each value
below is the median of two timed runs after one warmup. The full profile uses
five warmups and ten timed runs at the release workload sizes.

| Case | Backend | Host-boundary median (s) | p95 (s) | Resident median (s) | Peak allocator bytes |
|---|---:|---:|---:|---:|---:|
| float32 matmul 64 | base | 0.001014 | 0.001095 | 0.000439 | 0 |
| float32 matmul 64 | native | 0.004161 | 0.004596 | 0.000481 | 49,152 |
| float32 matmul 64 | torch | 0.004318 | 0.005300 | 0.000517 | 8,634,880 |
| float64 matmul 64 | base | 0.001034 | 0.001577 | 0.000351 | 0 |
| float64 matmul 64 | native | 0.002754 | 0.002887 | 0.000368 | 98,304 |
| float64 matmul 64 | torch | 0.003872 | 0.003972 | 0.001165 | 8,749,568 |
| dense PCA-kNN 128 x 16 | base | 0.009648 | 0.009912 | not separable | 0 |
| dense PCA-kNN 128 x 16 | native | 0.025820 | 0.026061 | not separable | 6,663,164 |
| dense PCA-kNN 128 x 16 | torch | 0.037548 | 0.037867 | not separable | 15,344,128 |
| sparse PCA-kNN 128 x 16 | base | 0.010294 | 0.010306 | 0.010100 | 0 |
| sparse PCA-kNN 128 x 16 | native | 0.025686 | 0.025850 | 0.027977 | 6,672,868 |
| sparse PCA-kNN 128 x 16 | torch | 0.045912 | 0.048574 | 0.041613 | 15,460,352 |

The smoke run measured a 1.742-second first diagnostics/self-test boundary.
This initialization measurement excludes R process startup and package
installation. The installed package footprints were:

- `cudaverse`: 2,164,157 bytes;
- optional `torch`: 7,367,799,444 bytes; and
- CUDA runtime bundled by `cudaverse`: 0 bytes.

At these small sizes, base R was faster at every host boundary. That is the
expected dispatch/transfer-overhead regime and is evidence against a universal
GPU-speed claim. Resident matmul narrows the difference substantially. Whether
native becomes advantageous at 256, 1024, or 4096 and for the larger PCA-kNN
workloads remains a question for the full retained report.
