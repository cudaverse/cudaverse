# CP-06 full benchmark evidence

- Schema: `cudaverse-benchmark/1`
- Profile: `full`
- Source commit: `42d6ab4c02c4ab20b807fe858043e8edb203e626`
- Source tracked dirty: `false`
- Report complete: `true`
- Report SHA-256: `9363aba0c4f5af0ee7ef84648bc2d6bec2177c7deccfd445454f9f7bdda43e93`
- Report generated: 2026-08-11 04:46:46 UTC
- Hardware: NVIDIA RTX 2000 Ada Generation, 595.97, 16380, 8.9
- R: R version 4.6.0 (2026-04-24 ucrt)
- cudaverse: `0.2.0.9000`
- torch: `0.17.0.9000`
- Stage sampling: pipeline stages are collected from the same synchronized timed host-boundary runs
- Memory sampling: one separate instrumented execution after timing; allocator tracking is excluded from retained timing samples

## Installed footprint

| Component | Installed bytes |
|---|---:|
| cudaverse | 1,125,384 |
| optional torch | 7,367,799,444 |
| CUDA runtime bundled by cudaverse | 0 |

## Matrix multiplication

| Case | Backend | Host median (s) | Host p95 (s) | Resident median (s) | Resident p95 (s) | Peak MiB | Max relative error | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul-float32-256 | base | 0.005227 | 0.005502 | 0.004459 | 0.004708 | 0.00 | 4e-08 | pass |
| matmul-float32-256 | native | 0.003668 | 0.003820 | 0.000312 | 0.000346 | 0.75 | 3.87e-07 | pass |
| matmul-float32-256 | torch | 0.004955 | 0.006158 | 0.000989 | 0.001136 | 9.88 | 3.87e-07 | pass |
| matmul-float64-256 | base | 0.004792 | 0.005134 | 0.004515 | 0.005006 | 0.00 | 0 | pass |
| matmul-float64-256 | native | 0.007649 | 0.009986 | 0.003916 | 0.004365 | 1.50 | 5.44e-16 | pass |
| matmul-float64-256 | torch | 0.008302 | 0.009592 | 0.004565 | 0.008082 | 11.63 | 5.44e-16 | pass |
| matmul-float32-1024 | base | 0.281448 | 0.289084 | 0.264154 | 0.271441 | 0.00 | 3.4e-08 | pass |
| matmul-float32-1024 | native | 0.031593 | 0.033602 | 0.005238 | 0.035133 | 12.00 | 2.8e-07 | pass |
| matmul-float32-1024 | torch | 0.044614 | 0.047763 | 0.006590 | 0.008874 | 36.13 | 2.8e-07 | pass |
| matmul-float64-1024 | base | 0.268047 | 0.274514 | 0.264391 | 0.270000 | 0.00 | 0 | pass |
| matmul-float64-1024 | native | 0.063690 | 0.077045 | 0.032530 | 0.034250 | 24.00 | 6.38e-16 | pass |
| matmul-float64-1024 | torch | 0.076237 | 0.086943 | 0.051405 | 0.059181 | 64.13 | 6.38e-16 | pass |
| matmul-float32-4096 | base | 27.945522 | 28.435725 | 27.063568 | 27.581672 | 0.00 | 3.91e-08 | pass |
| matmul-float32-4096 | native | 0.477618 | 0.626013 | 0.054252 | 0.068639 | 192.00 | 2.64e-06 | pass |
| matmul-float32-4096 | torch | 0.721094 | 0.847996 | 0.018857 | 0.019012 | 456.13 | 2.64e-06 | pass |
| matmul-float64-4096 | base | 27.627549 | 28.798226 | 27.190571 | 28.035418 | 0.00 | 0 | pass |
| matmul-float64-4096 | native | 0.810233 | 0.811781 | 0.717985 | 0.718059 | 384.00 | 9.23e-16 | pass |
| matmul-float64-4096 | torch | 1.095621 | 1.119507 | 0.715108 | 0.715201 | 904.13 | 9.23e-16 | pass |

## PCA and exact kNN pipelines

| Case | Backend | Host median (s) | Host p95 (s) | Resident continuation (s) | PCA stage (s) | kNN stage (s) | Peak MiB | Validation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense-1000x50 | base | 0.071142 | 0.075176 | not_separable | 0.004221 | 0.066461 | 0.00 | self-reference; pass |
| dense-1000x50 | native | 0.071041 | 0.072096 | not_separable | 0.036329 | 0.034666 | 9.26 | projector 4.72e-15; reconstruction 1.21e-14; kNN distance 1.09e-14; indices exact; pass |
| dense-1000x50 | torch | 0.177846 | 0.194183 | not_separable | 0.118603 | 0.058004 | 904.13 | projector 2.14e-14; reconstruction 4.19e-14; kNN distance 3.99e-14; indices exact; pass |
| dense-10000x100 | base | 10.238852 | 10.400593 | not_separable | 0.116703 | 10.120480 | 0.00 | self-reference; pass |
| dense-10000x100 | native | 0.202287 | 0.208085 | not_separable | 0.050408 | 0.152117 | 73.90 | projector 3.77e-14; reconstruction 9.91e-14; kNN distance 8.9e-14; indices exact; pass |
| dense-10000x100 | torch | 5.023404 | 6.620356 | not_separable | 0.272342 | 4.750463 | 904.13 | projector 6.96e-14; reconstruction 2.91e-13; kNN distance 2.44e-13; indices exact; pass |
| dense-50000x128 | base | 222.207632 | 285.474343 | not_separable | 1.311275 | 220.934222 | 0.00 | self-reference; pass |
| dense-50000x128 | native | 2.918310 | 2.929822 | not_separable | 0.209803 | 2.705260 | 440.97 | projector 1.72e-13; reconstruction 9.34e-13; kNN distance 9.22e-13; indices exact; pass |
| dense-50000x128 | torch | 173.604715 | 181.530578 | not_separable | 0.463083 | 173.248239 | 904.13 | projector 7.01e-13; reconstruction 4.61e-12; kNN distance 5.31e-12; indices exact; pass |
| sparse-1000x50@0.10 | base | 0.121018 | 0.134893 | 0.110640 | 0.008993 | 0.110000 | 0.00 | self-reference; pass |
| sparse-1000x50@0.10 | native | 0.097547 | 0.099647 | 0.095732 | 0.043612 | 0.048648 | 9.41 | projector 5.75e-14; reconstruction 6.65e-14; kNN distance 4.31e-14; indices exact; pass |
| sparse-1000x50@0.10 | torch | 0.201015 | 0.253159 | 0.193634 | 0.111617 | 0.080424 | 904.13 | projector 7.19e-14; reconstruction 1.04e-13; kNN distance 1.34e-13; indices exact; pass |
| sparse-10000x100@0.03 | base | 14.874136 | 15.649816 | 14.316848 | 0.146221 | 14.696792 | 0.00 | self-reference; pass |
| sparse-10000x100@0.03 | native | 0.238842 | 0.248709 | 0.232358 | 0.047101 | 0.186187 | 74.89 | projector 8.39e-13; reconstruction 1.26e-12; kNN distance 1.24e-12; indices exact; pass |
| sparse-10000x100@0.03 | torch | 7.406586 | 9.730072 | 6.562043 | 0.271465 | 7.111631 | 904.13 | projector 8.36e-13; reconstruction 1.27e-12; kNN distance 1.28e-12; indices exact; pass |
| sparse-50000x128@0.01 | base | 306.205827 | 340.882582 | 302.657116 | 1.786879 | 304.044213 | 0.00 | self-reference; pass |
| sparse-50000x128@0.01 | native | 4.901002 | 5.482177 | 5.048826 | 0.165130 | 4.708228 | 444.41 | projector 8.7e-12; reconstruction 1.49e-11; kNN distance 3.14e-11; indices exact; pass |
| sparse-50000x128@0.01 | torch | 156.881812 | 168.954545 | 153.214256 | 0.237781 | 156.613151 | 904.13 | projector 7.91e-12; reconstruction 1.4e-11; kNN distance 2.76e-11; indices exact; pass |

## Workload-specific observations

- `matmul-float32-256`: native had the numerically lowest host-boundary median; native median was 29.8% lower than base (ratio 1.43x), and native median was 26.0% lower than torch (ratio 1.35x).
  Resident matmul: native median was 68.5% lower than torch (ratio 3.17x).
- `matmul-float64-256`: base had the numerically lowest host-boundary median; native median was 59.6% higher than base (ratio 1.60x), and native median was 7.9% lower than torch (ratio 1.09x).
  Resident matmul: native median was 14.2% lower than torch (ratio 1.17x).
- `matmul-float32-1024`: native had the numerically lowest host-boundary median; native median was 88.8% lower than base (ratio 8.91x), and native median was 29.2% lower than torch (ratio 1.41x).
  Resident matmul: native median was 20.5% lower than torch (ratio 1.26x).
- `matmul-float64-1024`: native had the numerically lowest host-boundary median; native median was 76.2% lower than base (ratio 4.21x), and native median was 16.5% lower than torch (ratio 1.20x).
  Resident matmul: native median was 36.7% lower than torch (ratio 1.58x).
- `matmul-float32-4096`: native had the numerically lowest host-boundary median; native median was 98.3% lower than base (ratio 58.51x), and native median was 33.8% lower than torch (ratio 1.51x).
  Resident matmul: native median was 187.7% higher than torch (ratio 2.88x).
- `matmul-float64-4096`: native had the numerically lowest host-boundary median; native median was 97.1% lower than base (ratio 34.10x), and native median was 26.0% lower than torch (ratio 1.35x).
  Resident matmul: native median was 0.4% higher than torch (ratio 1.00x).
- `dense-1000x50`: native had the numerically lowest host-boundary median; native median was 0.1% lower than base (ratio 1.00x), and native median was 60.1% lower than torch (ratio 2.50x).
- `dense-10000x100`: native had the numerically lowest host-boundary median; native median was 98.0% lower than base (ratio 50.62x), and native median was 96.0% lower than torch (ratio 24.83x).
- `dense-50000x128`: native had the numerically lowest host-boundary median; native median was 98.7% lower than base (ratio 76.14x), and native median was 98.3% lower than torch (ratio 59.49x).
- `sparse-1000x50@0.10`: native had the numerically lowest host-boundary median; native median was 19.4% lower than base (ratio 1.24x), and native median was 51.5% lower than torch (ratio 2.06x).
- `sparse-10000x100@0.03`: native had the numerically lowest host-boundary median; native median was 98.4% lower than base (ratio 62.28x), and native median was 96.8% lower than torch (ratio 31.01x).
- `sparse-50000x128@0.01`: native had the numerically lowest host-boundary median; native median was 98.4% lower than base (ratio 62.48x), and native median was 96.9% lower than torch (ratio 32.01x).

## Interpretation boundaries

- These measurements describe one exact source commit on one RTX 2000 Ada system. They do not support a universal GPU speed claim.
- Ratios compare ten-run sample medians descriptively. They are not confidence intervals or statistical significance tests; small differences may be measurement noise.
- Host-boundary and resident timings answer different questions. Dense PCA upload is internal to the public boundary and is therefore reported as not separable.
- Peak memory is reported with its backend-specific allocator source in the machine-readable report; torch uses a session high-water source when its R API cannot reset a peak counter.
- Pipeline base results establish the numerical reference. Native and torch must match PCA projector/reconstruction, exact kNN indices, and distance tolerances before their timings are accepted.
