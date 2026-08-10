# cudaverse 0.2 backend capability audit

## Baseline

- development baseline: `6ab57469e7250fa73a3130f3b356934e34e2e0ca`;
- frozen `main` and `release/cran-0.1.0`:
  `59e15c8c5a56d26e09a594886c875b1b8249f6f9`;
- package version: `0.2.0.9000`;
- exported functions: 38;
- fast local suite: pass, with 22 hardware-only cases skipped by the explicit
  `CUDAVERSE_NATIVE_TESTS` gate.

The authoritative row-per-export matrix is
[`inst/reports/backend-capability-matrix.csv`](../inst/reports/backend-capability-matrix.csv).
`tools/check-capability-matrix.R` and the package test suite require it to
cover every `NAMESPACE` export exactly once.

## Status vocabulary

| Status | Meaning |
|---|---|
| `direct` | The main numerical operation is implemented by that backend. |
| `hybrid` | Meaningful stages are split between that backend and CPU code. |
| `cpu_only` | The operation deliberately runs on CPU for this 0.2 surface. |
| `metadata` | The function validates or returns metadata without a new numerical kernel. |
| `probe` | The function diagnoses or selects a backend. |
| `host` | The function explicitly materializes a host R object. |

`direct` describes execution, not result residency. The separate
`result_location` column distinguishes host results, device-resident results,
and backend-dependent results.

## Current contract summary

### Dense tensors

Construction, transfer, casting, arithmetic, broadcasting, reshape,
transpose, matrix multiplication, and reductions have direct base, torch, and
native implementations. Native reshape uses shared ownership. CUDA tensor
subsetting and replacement are intentionally outside the backend registry in
the current release and use a documented CPU round trip.

### Sparse matrices

Construction and sparse-dense multiplication exist for all three backends.
Native additionally provides device reductions, normalization, shared
ownership, device-to-host conversion, sparse PCA, and sparse kNN preparation.
The torch compatibility path computes sparse reductions from the stable host
metadata mirror and performs normalization on that mirror before rebuilding
CUDA sparse storage. Native sparse matrix-vector multiplication is a hybrid
API because the public return value is an R vector.

### Algorithms

SVD, dense PCA, and distance are direct on base, torch, and native. Native PCA
can retain score storage for a resident continuation. Native kNN performs
distance and stable top-k on the device; torch performs CUDA distance blocks
and stable CPU top-k. Sparse PCA and kNN are device-specialized only for
native. K-means is deliberately hybrid for CUDA backends because assignment
and centre updates remain CPU stages.

### Graphs and embeddings

Graph assembly uses `Matrix`; Louvain and Leiden use `igraph`. UMAP and t-SNE
use `uwot` and `Rtsne`. These functions preserve upstream device/provenance
metadata but are CPU-only in 0.2. Diffusion maps are hybrid: distance may use
torch or native while kernel construction and eigendecomposition remain CPU.
No new graph or embedding backend is required for the 0.2 candidate.

### Public S3 surface

`Ops`, `%*%`, transpose, reductions, reshape, and broadcasting use the tensor
backend. `as.array()`, `as.matrix()`, `to_cpu()`, and sparse conversion are
explicit host materializations. CUDA `[` and `[<-` currently round-trip
through base R. PCA prediction uses the selected algorithm backend; k-means
prediction is hybrid whenever distance uses CUDA.

## Checkpoint 2 resolution

1. **Capability semantics resolved.** Backend registration now validates
   capability declarations. Diagnostics expose human-readable `capabilities`
   separately from callable internal `operations`; the legacy diagnostic
   fields remain unchanged.
2. **Optional dispatch generalized.** Sparse PCA and kNN now select their
   specialized paths by registered operation rather than the literal backend
   name. Storage-specific ownership checks remain where the object contract is
   genuinely native-specific.
3. **Torch sparse provenance resolved.** CPU normalization and CUDA sparse
   reconstruction are separate `normalization` and `normalization_upload`
   stages.
4. **Known CPU boundaries are release limitations, not missing repositories.**
   Tensor subsetting/replacement, graph/community workflows, UMAP, t-SNE, the
   non-distance portion of diffusion maps, and k-means updates remain explicit
   CPU or hybrid paths for 0.2.

Findings 1-3 are covered by backend contract and sparse provenance tests.
Finding 4 remains an explicit 0.2 support boundary.
