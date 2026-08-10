# Native CUDA consolidation record

## Objective

Make `cudaverse` the single repository and R package for the general numerical
API and its lightweight native CUDA implementation. `cudacellr` remains the
only separate domain package. The `cudaverseCUDA` repository is retained until
the integrated implementation passes every replacement gate, then archived
with a redirect rather than deleted.

## Immutable baselines

- frozen `main` and `release/cran-0.1.0`: `59e15c8`;
- consolidation parent on `develop/native-cuda`: `7907bd9`;
- imported native source: `cudaverseCUDA` `38266cd`;
- pre-consolidation Phase 4 hardware evidence:
  `inst/reports/native/phase4-rtx2000.json`.

## Checkpoints

### 1. Baseline and core integration

- [x] Both source repositories were clean and synchronized before migration.
- [x] The original `cudaverse` suite passed before migration.
- [x] The original extension contract suite passed, with hardware tests gated.
- [x] Native R, C++17, PTX, tests, historical reports, and compliance evidence
  were copied into `cudaverse`.
- [x] Native registration is internal; `Suggests`, `Remotes`,
  `Additional_repositories`, `requireNamespace()`, and exported extension
  factory coupling were removed.
- [x] The integrated package compiled on local Windows/R 4.6 and the combined
  non-hardware test suite passed.
- [x] With explicit user-provided cuBLAS/cuSOLVER paths, the full native RTX
  test file passed from `cudaverse` without loading `cudaverseCUDA`.

### 2. Packaging, documentation, and supply chain

- [x] Native runtime loading remains lazy and no NVIDIA or LibTorch runtime is
  bundled.
- [x] The CycloneDX SBOM identifies `cudaverse` as the component and matches
  the checksum-pinned package-owned PTX.
- [x] The redistribution, SBOM, and historical Phase 4 evidence gates pass.
- [x] README, NEWS, backend documentation, and roadmap describe the integrated
  architecture.
- [x] A dedicated workflow covers CUDA 12.8.1 ABI/PTX reproducibility and
  Windows/Linux no-CUDA artifact installation.

### 3. Remaining validation

- [x] An isolated R library in which `cudaverseCUDA` was unavailable built,
  installed, loaded, diagnosed the integrated native backend, and executed a
  CPU tensor successfully.
- [x] Windows R 4.6.0 `R CMD check --as-cran --no-manual` completed with
  `0 ERROR / 0 WARNING / 1 NOTE`; the only NOTE was the expected development
  version/new-submission metadata. The source archive was 164,578 bytes,
  installed tree was 1,063,677 bytes, and source SHA-256 was
  `29b3d533b134e3e08f559f57b7a92ff14bd155de028070c9fa48c38c5999e49b`.
  The retained log is
  `.github/evidence/native-consolidation-windows-r46-check.log`.
- [x] The full integrated RTX report passed numerical parity, automatic
  selection, resident provenance, error recovery, interruption, dense and
  sparse 1,000-cycle lifetime, timing, peak-VRAM, and combined installed-size
  gates. Its SHA-256 is
  `13ed0e28c1337dbfd934c5b6dbb1a81a26f2b58e73b04194c63c2c964f5f2888`.
- [ ] Obtain green Windows, macOS, Ubuntu, R-devel, native integrity, and
  trusted RTX GitHub checks for the reviewed candidate.

### 4. Repository transition

- [ ] Merge the reviewed consolidation into `develop/native-cuda`.
- [ ] Add a migration notice to `cudaverseCUDA` and archive it on GitHub.
- [ ] Move its clean local checkout under `cudaverse/_archived`.
- [ ] Record final commits, checks, artifact checksums, sizes, and remaining
  release/CRAN risks here.

## Safety boundaries

This consolidation does not change or submit the frozen 0.1 CRAN candidate,
start `cudacellr` development, create a 0.2 tag or release, move an existing
tag, bundle an NVIDIA runtime, or remove the torch compatibility backend.
