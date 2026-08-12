# cudaverse 0.4 development roadmap

## Outcome and boundaries

The 0.4 line turns the validated lightweight CUDA core into a production-
oriented R experience: easier setup, fewer host round trips, stronger dense
and sparse workflows, and explicit limitations. It remains one repository and
one user package. Native CUDA stays optional, lazy, LibTorch-independent, and
safe to install without CUDA.

The exact 0.3 candidate is
`b0ab90f8547d42a5244963bc6b15d0ba223382ff`. Its compact 1.84 MiB evidence
bundle is retained outside the source tree at
`cudaverse/evidence/0.3.0.9000-b0ab90f`; manifest SHA-256 is
`858D2691D276CD26EA225B4B57BA2219644466181869977C1AC87400D9FCDFF1`.
`main` and `release/cran-0.1.0` remain frozen at
`59e15c8c5a56d26e09a594886c875b1b8249f6f9`. External release actions require
explicit maintainer approval.

## Audited 0.4 baseline

At the branch point, the package has 38 exports, 28 registered S3 methods,
179 test blocks, 163 tracked files, and a 1.48 MiB tracked source payload.
GitHub has no open issues. The 0.3 candidate passed Windows, macOS, Ubuntu,
R-devel, no-CUDA artifacts, CUDA 12.8.1 ABI/PTX, supply-chain, RTX 2000,
1,000-cycle lifecycle, documentation, and 12-case/36-result benchmark gates.

Intentional boundaries remain visible in
`inst/reports/backend-capability-matrix.csv`: graph construction and community
detection are CPU; UMAP and t-SNE are CPU; diffusion maps are hybrid; torch
exact-kNN and several sparse stages are hybrid; native sparse storage is
float64 until new evidence justifies float32.

## Ordered milestones

1. **Setup and diagnostics.** Add user-facing health, backend comparison, and
   actionable next steps without removing compatibility fields. Keep explicit
   CUDA strict and no-CUDA installation normal.
2. **Tensor residency.** Audit and remove avoidable transfers in indexing,
   replacement, views, shape operations, casting, reductions, dimnames, and
   printing. Preserve ownership, interruption, and provenance.
3. **Dense numerical core.** Improve resident reductions, PCA/SVD, distance,
   stable top-k/kNN, and k-means. Add measured batching/chunking for VRAM
   pressure rather than hidden host fallback.
4. **Sparse numerical core.** Optimize COO/CSR conversion, transpose,
   normalization, multiplication, reductions, PCA/kNN preparation, Matrix
   interoperability, and memory. Reconsider float32 only through written
   accuracy, footprint, and performance evidence.
5. **Higher-level workflows.** Audit graph, clustering, embeddings, and
   SingleCellExperiment paths; accelerate only justified stages and document
   intentional CPU/hybrid work.
6. **Reliability and observability.** Expand conformance, invalid-input, OOM,
   injected-error, interruption, shared-view, backend-reuse, session, VRAM,
   and 1,000-cycle lifecycle gates.
7. **Performance and packaging.** Track cold/warm time, transfer volume, peak
   VRAM, allocator behavior, installed size, PTX/ABI reproducibility, SBOM,
   licenses, cross-platform checks, references, tutorials, and pkgdown.
8. **Exact candidate review.** Bind one clean 0.4 commit to matching source,
   CI, artifacts, RTX, benchmarks, supply-chain, documentation, and decision
   evidence, then stop before external release.

## Validation gates

- Integer and index results are exact. Float32 uses `rtol = 1e-5` and
  `atol = 1e-6`; float64 uses `rtol = 1e-10` normally and `1e-8` for heavy
  numerical operations. PCA uses projector/reconstruction/subspace checks;
  kNN ties retain original row order.
- `device = "cuda"` never silently falls back. Automatic selection records the
  actual backend and reason. CPU-only hosts install, load, check, and run.
- Native dense and sparse 1,000-cycle post-cleanup difference is at most
  1 MiB with no double-free or invalid shared view.
- Every export has tested and documented base, torch, and native behavior,
  including explicit unsupported or hybrid boundaries.

Each checkpoint records scope, evidence, and unresolved risk in the internal
0.4 checkpoint log. A merged checkpoint is progress, not a release decision.
