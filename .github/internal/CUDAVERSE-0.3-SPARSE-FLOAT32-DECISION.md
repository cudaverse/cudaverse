# Internal cudaverse 0.3 sparse float32 decision

## Decision

Native float32 sparse storage is **evidence-deferred** for the 0.3 candidate.
The supported sparse numerical surface remains float64 across base, torch, and
native backends. This is a bounded scope decision, not a claim that float32
sparse storage would never be useful.

## Current evidence

- `cuda_sparse()` has no dtype argument. It validates numeric input, converts
  `Matrix` inputs through `dMatrix`, and exposes numeric values with the same
  float64 contract on every backend.
- Native sparse storage copies values as `double`. Its transpose, row/column
  reductions, normalization, sparse-dense multiplication, and sparse-to-dense
  kernels are explicitly `f64`.
- The retained sparse Phase 3 RTX 2000 report already proves float64 parity,
  resident normalization -> PCA -> exact kNN, structured recovery, and a
  1,000-cycle lifecycle result. At `10000 x 128 @ 0.01`, native completed the
  full workflow in a 0.210-second median with 89.8 MiB peak backend allocation.
- There is no retained float32 sparse prototype showing an accuracy, memory,
  or elapsed-time advantage. Adding it now would duplicate storage and kernel
  paths, widen dtype-promotion rules, and enlarge the conformance/lifecycle
  matrix without evidence that users benefit.

## Re-entry gate

Reconsider float32 sparse storage after the 0.3 candidate only when a bounded
prototype can run without changing the public `device` contract and proves all
of the following on the same RTX evidence source:

1. base/torch/native parity within `rtol = 1e-5` and `atol = 1e-6`, including
   normalization, PCA projector/reconstruction, exact kNN indices, dimnames,
   transpose, reductions, and sparse-dense multiplication;
2. at least 25% lower operation-owned peak allocation on both the `10000 x
   100` and `50000 x 128` sparse benchmark cases;
3. at least 15% lower native full-workflow median on one of those cases, with
   no greater than 10% regression on the other;
4. no host round trip added between normalization, PCA scores, distance, and
   top-k, with complete `cudaverse-stage/1` provenance;
5. 1,000 allocate/transfer/operate/free cycles, injected error, interruption,
   shared-owner release, and backend reuse remain within the 1 MiB cleanup
   ceiling; and
6. installed artifacts remain lightweight and all CPU/no-CUDA checks remain
   independent of a CUDA runtime.

If the prototype misses these gates, float64 remains the sparse contract. This
keeps 0.3 focused on a smaller, better-proven surface instead of treating an
additional dtype as progress by itself.
