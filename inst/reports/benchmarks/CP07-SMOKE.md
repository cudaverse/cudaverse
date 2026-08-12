# cudaverse smoke benchmark evidence

- Schema: `cudaverse-benchmark/1`
- Profile: `smoke`
- Source commit: `d1d4e83686d71431dc0c6a46998ca48a9e926f30`
- Source tracked dirty: `false`
- Report complete: `true`
- Report SHA-256: `d79bb27942994e6469d51eb0e346bcb1be6305d83cf562643f70dcb5e6bdba87`
- Report generated: 2026-08-12 07:52:20 UTC
- Hardware: NVIDIA RTX 2000 Ada Generation, 595.97, 16380, 8.9
- R: R version 4.6.0 (2026-04-24 ucrt)
- cudaverse: `0.4.0.9000`
- torch: `0.17.0.9000`
- Stage sampling: pipeline stages are collected from the same synchronized timed host-boundary runs
- Memory sampling: one separate instrumented execution after timing; allocator tracking is excluded from retained timing samples

## Installed footprint

| Component | Installed bytes |
|---|---:|
| cudaverse | 1,384,393 |
| optional torch | 7,367,799,444 |
| CUDA runtime bundled by cudaverse | 0 |

## Matrix multiplication

| Case | Backend | Host median (s) | Host p95 (s) | Resident median (s) | Resident p95 (s) | Peak MiB | Max relative error | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul-float32-64 | base | 0.002805 | 0.003778 | 0.000437 | 0.000605 | 0.00 | 4.08e-08 | pass |
| matmul-float32-64 | native | 0.005985 | 0.008448 | 0.000390 | 0.000452 | 0.05 | 1.41e-07 | pass |
| matmul-float32-64 | torch | 0.004301 | 0.004578 | 0.000497 | 0.000602 | 0.05 | 1.41e-07 | pass |
| matmul-float64-64 | base | 0.000504 | 0.000523 | 0.000385 | 0.000460 | 0.00 | 0 | pass |
| matmul-float64-64 | native | 0.003255 | 0.003296 | 0.000324 | 0.000343 | 0.09 | 2.31e-16 | pass |
| matmul-float64-64 | torch | 0.003721 | 0.003801 | 0.000770 | 0.001190 | 0.09 | 2.31e-16 | pass |

## PCA and exact kNN pipelines

| Case | Backend | Host median (s) | Host p95 (s) | Resident continuation (s) | PCA stage (s) | kNN stage (s) | Peak MiB | Validation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense-128x16 | base | 0.004795 | 0.006271 | not_separable | 0.001271 | 0.003497 | 0.00 | self-reference; pass |
| dense-128x16 | native | 0.015499 | 0.015784 | not_separable | 0.011966 | 0.003501 | 6.35 | projector 1.94e-15; reconstruction 1.07e-14; kNN distance 6.11e-15; indices exact; pass |
| dense-128x16 | torch | 0.024230 | 0.026148 | not_separable | 0.017567 | 0.006637 | 6.08 | projector 9.77e-15; reconstruction 1.64e-14; kNN distance 1.49e-14; indices exact; pass |
| sparse-128x16@0.10 | base | 0.005107 | 0.005315 | 0.004508 | 0.001203 | 0.002572 | 0.00 | self-reference; pass |
| sparse-128x16@0.10 | native | 0.023425 | 0.024655 | 0.046920 | 0.014454 | 0.004604 | 6.36 | projector 1.73e-14; reconstruction 2.79e-14; kNN distance 6.11e-13; indices exact; pass |
| sparse-128x16@0.10 | torch | 0.031266 | 0.032440 | 0.046893 | 0.020212 | 0.005133 | 6.33 | projector 1.37e-14; reconstruction 2.33e-14; kNN distance 1.03e-13; indices exact; pass |

## Workload-specific observations

- `matmul-float32-64`: base had the numerically lowest host-boundary median; native median was 113.4% higher than base (ratio 2.13x), and native median was 39.1% higher than torch (ratio 1.39x).
  Resident matmul: native median was 21.5% lower than torch (ratio 1.27x).
- `matmul-float64-64`: base had the numerically lowest host-boundary median; native median was 545.3% higher than base (ratio 6.45x), and native median was 12.5% lower than torch (ratio 1.14x).
  Resident matmul: native median was 58.0% lower than torch (ratio 2.38x).
- `dense-128x16`: base had the numerically lowest host-boundary median; native median was 223.2% higher than base (ratio 3.23x), and native median was 36.0% lower than torch (ratio 1.56x).
- `sparse-128x16@0.10`: base had the numerically lowest host-boundary median; native median was 358.7% higher than base (ratio 4.59x), and native median was 25.1% lower than torch (ratio 1.33x).

## Interpretation boundaries

- These measurements describe one exact source commit on one RTX 2000 Ada system. They do not support a universal GPU speed claim.
- Ratios compare ten-run sample medians descriptively. They are not confidence intervals or statistical significance tests; small differences may be measurement noise.
- Host-boundary and resident timings answer different questions. Dense PCA upload is internal to the public boundary and is therefore reported as not separable.
- Peak memory is reported with its backend-specific allocator source in the machine-readable report; torch uses a session high-water source when its R API cannot reset a peak counter.
- Pipeline base results establish the numerical reference. Native and torch must match PCA projector/reconstruction, exact kNN indices, and distance tolerances before their timings are accepted.
