# cudaverse full benchmark evidence

- Schema: `cudaverse-benchmark/1`
- Profile: `full`
- Source commit: `deb77127eebf68e0c6f788e9e10c40c5cb8dacfc`
- Source tracked dirty: `false`
- Report complete: `true`
- Report SHA-256: `707aecbd0d72a5b1e1bd209a0892b555c41771fe9e3870f5414b7613650c380a`
- Report generated: 2026-08-12 18:27:32 UTC
- Hardware: NVIDIA RTX 2000 Ada Generation, 595.97, 16380, 8.9
- R: R version 4.6.0 (2026-04-24 ucrt)
- cudaverse: `0.4.0.9000`
- torch: `0.17.0.9000`
- Stage sampling: pipeline stages are collected from the same synchronized timed host-boundary runs
- Memory sampling: one separate instrumented execution after timing; allocator tracking is excluded from retained timing samples

## Installed footprint

| Component | Installed bytes |
|---|---:|
| cudaverse | 1,447,216 |
| optional torch | 7,367,799,444 |
| CUDA runtime bundled by cudaverse | 0 |

## Matrix multiplication

| Case | Backend | Host median (s) | Host p95 (s) | Resident median (s) | Resident p95 (s) | Peak MiB | Max relative error | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul-float32-256 | base | 0.006096 | 0.011986 | 0.004921 | 0.009245 | 0.00 | 4e-08 | pass |
| matmul-float32-256 | native | 0.005462 | 0.007928 | 0.000510 | 0.006860 | 0.75 | 3.87e-07 | pass |
| matmul-float32-256 | torch | 0.009550 | 0.015407 | 0.000969 | 0.008798 | 0.75 | 3.87e-07 | pass |
| matmul-float64-256 | base | 0.005033 | 0.009780 | 0.004484 | 0.008785 | 0.00 | 0 | pass |
| matmul-float64-256 | native | 0.004939 | 0.028728 | 0.001509 | 0.008838 | 1.50 | 5.44e-16 | pass |
| matmul-float64-256 | torch | 0.006293 | 0.006582 | 0.003213 | 0.022180 | 1.50 | 5.44e-16 | pass |
| matmul-float32-1024 | base | 0.332366 | 0.444637 | 0.310773 | 0.354043 | 0.00 | 3.4e-08 | pass |
| matmul-float32-1024 | native | 0.024284 | 0.026370 | 0.001998 | 0.021575 | 12.00 | 2.8e-07 | pass |
| matmul-float32-1024 | torch | 0.038870 | 0.224033 | 0.001127 | 0.005731 | 12.00 | 2.8e-07 | pass |
| matmul-float64-1024 | base | 0.344592 | 0.395868 | 0.342931 | 0.426682 | 0.00 | 0 | pass |
| matmul-float64-1024 | native | 0.028283 | 0.033589 | 0.028241 | 0.031772 | 24.00 | 6.38e-16 | pass |
| matmul-float64-1024 | torch | 0.053046 | 0.070464 | 0.033364 | 0.104593 | 24.00 | 6.38e-16 | pass |
| matmul-float32-4096 | base | 36.172490 | 37.929798 | 35.982862 | 38.007138 | 0.00 | 3.91e-08 | pass |
| matmul-float32-4096 | native | 0.580275 | 0.634443 | 0.031866 | 0.040795 | 192.00 | 2.64e-06 | pass |
| matmul-float32-4096 | torch | 0.813368 | 0.963128 | 0.018829 | 0.019328 | 192.00 | 2.64e-06 | pass |
| matmul-float64-4096 | base | 36.580272 | 39.574223 | 36.032241 | 37.404898 | 0.00 | 0 | pass |
| matmul-float64-4096 | native | 0.838240 | 0.946894 | 0.796963 | 0.980715 | 384.00 | 9.23e-16 | pass |
| matmul-float64-4096 | torch | 1.217972 | 1.273276 | 0.803599 | 0.855152 | 384.00 | 9.23e-16 | pass |

## PCA and exact kNN pipelines

| Case | Backend | Host median (s) | Host p95 (s) | Resident continuation (s) | PCA stage (s) | kNN stage (s) | Peak MiB | Validation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense-1000x50 | base | 0.089926 | 0.100306 | not_separable | 0.004436 | 0.083776 | 0.00 | self-reference; pass |
| dense-1000x50 | native | 0.026722 | 0.044492 | not_separable | 0.014916 | 0.008075 | 9.26 | projector 4.72e-15; reconstruction 1.21e-14; kNN distance 1.09e-14; indices exact; pass |
| dense-1000x50 | torch | 0.102586 | 0.190339 | not_separable | 0.046901 | 0.053072 | 896.00 | projector 2.14e-14; reconstruction 4.19e-14; kNN distance 3.99e-14; indices exact; pass |
| dense-10000x100 | base | 11.914605 | 12.335337 | not_separable | 0.127417 | 11.794610 | 0.00 | self-reference; pass |
| dense-10000x100 | native | 0.274680 | 0.291187 | not_separable | 0.056904 | 0.215931 | 73.90 | projector 3.77e-14; reconstruction 9.91e-14; kNN distance 8.9e-14; indices exact; pass |
| dense-10000x100 | torch | 6.094423 | 7.723461 | not_separable | 0.219539 | 5.870918 | 896.00 | projector 6.96e-14; reconstruction 2.91e-13; kNN distance 2.44e-13; indices exact; pass |
| dense-50000x128 | base | 668.417738 | 696.146041 | not_separable | 3.737343 | 664.678560 | 0.00 | self-reference; pass |
| dense-50000x128 | native | 2.806777 | 2.831888 | not_separable | 0.150792 | 2.653010 | 440.97 | projector 1.72e-13; reconstruction 9.34e-13; kNN distance 9.22e-13; indices exact; pass |
| dense-50000x128 | torch | 126.552305 | 127.947779 | not_separable | 0.351854 | 126.323814 | 392.27 | projector 7.01e-13; reconstruction 4.61e-12; kNN distance 5.31e-12; indices exact; pass |
| sparse-1000x50@0.10 | base | 0.077976 | 0.082338 | 0.072926 | 0.004234 | 0.072038 | 0.00 | self-reference; pass |
| sparse-1000x50@0.10 | native | 0.092981 | 0.096346 | 0.087866 | 0.043176 | 0.044588 | 9.37 | projector 5.75e-14; reconstruction 6.65e-14; kNN distance 4.31e-14; indices exact; pass |
| sparse-1000x50@0.10 | torch | 0.181455 | 0.194876 | 0.174612 | 0.115325 | 0.057666 | 391.83 | projector 7.19e-14; reconstruction 1.04e-13; kNN distance 1.34e-13; indices exact; pass |
| sparse-10000x100@0.03 | base | 10.265072 | 10.401679 | 9.911606 | 0.108342 | 10.148979 | 0.00 | self-reference; pass |
| sparse-10000x100@0.03 | native | 0.202858 | 0.207303 | 0.200089 | 0.041823 | 0.152287 | 74.62 | projector 8.39e-13; reconstruction 1.26e-12; kNN distance 1.24e-12; indices exact; pass |
| sparse-10000x100@0.03 | torch | 5.197889 | 6.892998 | 5.207299 | 0.298722 | 4.926844 | 389.54 | projector 8.36e-13; reconstruction 1.27e-12; kNN distance 1.28e-12; indices exact; pass |
| sparse-50000x128@0.01 | base | 217.632367 | 219.815376 | 210.974479 | 1.136508 | 216.457614 | 0.00 | self-reference; pass |
| sparse-50000x128@0.01 | native | 2.799527 | 2.815579 | 2.805922 | 0.127005 | 2.646970 | 443.45 | projector 8.7e-12; reconstruction 1.49e-11; kNN distance 3.14e-11; indices exact; pass |
| sparse-50000x128@0.01 | torch | 126.738861 | 129.178788 | 127.808142 | 0.346953 | 126.383013 | 434.66 | projector 7.91e-12; reconstruction 1.4e-11; kNN distance 2.76e-11; indices exact; pass |

## Workload-specific observations

- `matmul-float32-256`: native had the numerically lowest host-boundary median; native median was 10.4% lower than base (ratio 1.12x), and native median was 42.8% lower than torch (ratio 1.75x).
  Resident matmul: native median was 47.4% lower than torch (ratio 1.90x).
- `matmul-float64-256`: native had the numerically lowest host-boundary median; native median was 1.9% lower than base (ratio 1.02x), and native median was 21.5% lower than torch (ratio 1.27x).
  Resident matmul: native median was 53.0% lower than torch (ratio 2.13x).
- `matmul-float32-1024`: native had the numerically lowest host-boundary median; native median was 92.7% lower than base (ratio 13.69x), and native median was 37.5% lower than torch (ratio 1.60x).
  Resident matmul: native median was 77.2% higher than torch (ratio 1.77x).
- `matmul-float64-1024`: native had the numerically lowest host-boundary median; native median was 91.8% lower than base (ratio 12.18x), and native median was 46.7% lower than torch (ratio 1.88x).
  Resident matmul: native median was 15.4% lower than torch (ratio 1.18x).
- `matmul-float32-4096`: native had the numerically lowest host-boundary median; native median was 98.4% lower than base (ratio 62.34x), and native median was 28.7% lower than torch (ratio 1.40x).
  Resident matmul: native median was 69.2% higher than torch (ratio 1.69x).
- `matmul-float64-4096`: native had the numerically lowest host-boundary median; native median was 97.7% lower than base (ratio 43.64x), and native median was 31.2% lower than torch (ratio 1.45x).
  Resident matmul: native median was 0.8% lower than torch (ratio 1.01x).
- `dense-1000x50`: native had the numerically lowest host-boundary median; native median was 70.3% lower than base (ratio 3.37x), and native median was 74.0% lower than torch (ratio 3.84x).
- `dense-10000x100`: native had the numerically lowest host-boundary median; native median was 97.7% lower than base (ratio 43.38x), and native median was 95.5% lower than torch (ratio 22.19x).
- `dense-50000x128`: native had the numerically lowest host-boundary median; native median was 99.6% lower than base (ratio 238.14x), and native median was 97.8% lower than torch (ratio 45.09x).
- `sparse-1000x50@0.10`: base had the numerically lowest host-boundary median; native median was 19.2% higher than base (ratio 1.19x), and native median was 48.8% lower than torch (ratio 1.95x).
- `sparse-10000x100@0.03`: native had the numerically lowest host-boundary median; native median was 98.0% lower than base (ratio 50.60x), and native median was 96.1% lower than torch (ratio 25.62x).
- `sparse-50000x128@0.01`: native had the numerically lowest host-boundary median; native median was 98.7% lower than base (ratio 77.74x), and native median was 97.8% lower than torch (ratio 45.27x).

## Interpretation boundaries

- These measurements describe one exact source commit on one RTX 2000 Ada system. They do not support a universal GPU speed claim.
- Ratios compare ten-run sample medians descriptively. They are not confidence intervals or statistical significance tests; small differences may be measurement noise.
- Host-boundary and resident timings answer different questions. Dense PCA upload is internal to the public boundary and is therefore reported as not separable.
- Peak memory is reported with its backend-specific allocator source in the machine-readable report; torch uses a session high-water source when its R API cannot reset a peak counter.
- Pipeline base results establish the numerical reference. Native and torch must match PCA projector/reconstruction, exact kNN indices, and distance tolerances before their timings are accepted.
