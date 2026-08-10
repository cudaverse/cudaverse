# cudaverse 0.3 checkpoint log

## Baseline audit

- Candidate line: `develop/native-cuda` at
  `1b1707366f06a9667063f1703f6470117d517320`.
- Frozen lines: `main` and `release/cran-0.1.0` at
  `59e15c8c5a56d26e09a594886c875b1b8249f6f9`.
- Surface: 38 exports, 27 S3 registrations, and a row-per-export backend
  capability matrix enforced by tests and `tools/check-capability-matrix.R`.
- Local baseline: complete CPU suite passed; native cases were correctly gated
  behind `CUDAVERSE_NATIVE_TESTS=true`.
- GitHub baseline: no open issues or pull requests; the preceding checkpoint
  had green Windows/macOS/Ubuntu/R-devel, CPU contract, pkgdown, supply-chain,
  artifact, CUDA ABI, and reproducible PTX checks.
- Payload: 123 tracked files and about 1.1 MB of tracked data, including about
  166 KB of PTX and 253 KB of retained reports. No CUDA Toolkit, NVIDIA runtime,
  LibTorch, or other large binary payload was present.

### Audit findings

1. CUDA tensor `[` and `[<-` were the remaining full-value host round trips in
   the dense object API.
2. Native PCA, distance, and stable top-k already retain their important
   intermediate buffers, but k-means and several graph/embedding stages remain
   explicitly hybrid or CPU-only.
3. Native sparse COO/CSR, normalization, reductions, multiplication, PCA, and
   kNN preparation exist; transpose and broader conformance coverage remain 0.3
   work.
4. Historical reports validate earlier hardware checkpoints but do not prove a
   future 0.3 candidate. Final evidence must be regenerated from one exact
   clean commit.

## CP-01: device-native tensor indexing

Status: implementation in progress on `agent/native-indexing`.

Scope:

- add registry operations for tensor gather and replacement;
- preserve ordinary R indexing, drop, dimnames, dtype, recycling, and
  last-write semantics for duplicate replacement indices;
- keep native tensor values and compatible tensor replacements on the GPU;
- retain explicit compatibility fallback for missing indices and backends that
  do not implement indexing;
- extend native self-test, parity tests, capabilities, documentation, and PTX.

Required evidence before merge:

- [x] native bridge compiles on local Windows without a CUDA Toolkit;
- [x] local CPU suite passes without weakening tests;
- [x] CUDA 12.8.1 PTX rebuilt by the pinned CI container; committed SHA-256
  `42f0bf069d4c4b5ec507658f43e01088391508f10e63358a2a4b68d0bd331bd4`
  is recorded in `SHA256SUMS` and the CycloneDX SBOM;
- [x] Windows, macOS, Ubuntu, R-devel, pkgdown, supply-chain, CPU contract,
  CUDA ABI/PTX, and Windows/Linux artifact CI passed on the checkpoint source;
- [x] exact clean commit `888aa9bdb0d9e6961c4f0808b24c83142fe86b87`
  passed the complete native test file on the RTX 2000 with no skipped cases,
  including integer/float32/float64 indexing and device-resident tensor-value
  replacement parity;
- [x] the same RTX run passed structured indexing error recovery and 1,000
  gather/scatter cycles with no tracked leak and at most 1 MiB whole-device
  post-cleanup difference;
- [x] the checkpoint source is committed and synchronized on PR 16 into
  `develop/native-cuda`; all non-hardware PR checks are green, and the exact
  source also passed the local RTX gate above.

Additional local packaging evidence: a full source build generated all three
vignettes and `R CMD check --no-manual` completed with `Status: OK` on R 4.6.0
for Windows.

Deferred beyond CP-01:

- missing-index gather remains a provenance-visible compatibility path;
- native sparse transpose and the broader export-parameterized conformance
  suite remain later milestones.
