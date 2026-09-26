# Native matmul precision comparison

Run `tools/run-matmul-precision-experiment.R` from the repository root with the
intended installed `cudaverse` first on R's library path. It compares public
`tensor_matmul()` calls using standard float32, explicitly permitted TF32, and
standard float64. Every mode receives the same float32-rounded inputs. This
isolates compute precision from initial input quantization.

The diagnostic profile uses the original 64-by-64 fixture, 256 and 1024 square
fixtures, and exact small-integer identity and odd rectangular controls. It
records one first sample, one warmup, and three measured samples per timing
scope. Resident matmul and complete host-to-device-to-host timings are separate
and synchronized. Output checks, downloads for resident validation, and garbage
collection occur outside the timer. The first sample follows initialization
and correctness probes; it is not a fresh-process CUDA cold start.

For example, in PowerShell after selecting the installed package and CUDA
library paths:

```powershell
Remove-Item Env:NVIDIA_TF32_OVERRIDE -ErrorAction SilentlyContinue
$env:CUDAVERSE_MATMUL_PRECISION_PROFILE = 'diagnostic'
$env:CUDAVERSE_MATMUL_PRECISION_OUTPUT = 'C:/evidence/precision-diagnostic.json'
Rscript tools/run-matmul-precision-experiment.R
```

The output file must be new and its parent directory must already exist.
Diagnostic runs may use a shared GPU or an installation without a verified
source manifest. Their reports label those limitations and always set
`retained_performance_eligible` to false. Do not publish speed claims from these
durations.

## Full retained profile

The full profile adds a 4096 square fixture and uses five warmups and ten
measured samples per scope and mode. It requires a clean checkout, a verified
installation of that source, and an idle GPU. The existing GPU guard checks
compute processes before and after every first, warmup, and measured sample;
there is no environment switch to disable this requirement.

Create a verified candidate installation from a clean checkout:

```text
Rscript tools/install-benchmark-candidate.R <new-candidate-directory>
```

Put `<new-candidate-directory>/lib` first on `R_LIBS_USER`, retaining a library
with `jsonlite` and `digest` as necessary. Set
`CUDAVERSE_BENCHMARK_INSTALL_MANIFEST` to the candidate's `manifest.json`.
Then run the driver with `CUDAVERSE_MATMUL_PRECISION_PROFILE=full` and a new
`CUDAVERSE_MATMUL_PRECISION_OUTPUT` path. The installation helper and manifest
check connect the source revision to the installed package payload, including
the native library. Matching package versions alone are insufficient.

Keep `NVIDIA_TF32_OVERRIDE` unset for this comparison. The separate standard
benchmark's optional **torch** backend requires a new process with
`NVIDIA_TF32_OVERRIDE=0` to exclude framework-level TF32. In this native
comparison, that setting disables the explicit TF32 request and the driver
fails with an explanation. Native `precision="standard"` and `precision="tf32"`
are per-call policies; the experiment checks that intervening TF32 calls leave
subsequent explicit standard and default results bitwise unchanged.

## Reading the report

Standard float32 uses `benchmark_float32_matmul_validation()` from
`tools/benchmark-validation.R`. TF32 uses the separate componentwise model in
`tools/matmul-precision-validation.R`: it includes nearest-rounded TF32 input
error, float32 accumulation error, CPU double reference error, and error in the
computed absolute-product sum. Both preserve the older fixed elementwise
tolerance as a diagnostic. Float64 uses its existing fixed tolerance. Exact
control cases must also match exactly. Shape, dimnames, dtype, native
provenance, repeated-sample values, and allocation cleanup are required.

The TF32 rule states its arithmetic assumptions and range limits. It is not a
universal PTX numerical guarantee: NVIDIA leaves TF32 MMA rounding, reduction
order, and subnormal handling unspecified. A TF32 request permits cuBLAS to use
Tensor Cores; the report does not claim that a particular hardware kernel was
selected. See the source links in the policy metadata.

Reports include raw samples, source and installed-binary hashes, policy hashes,
input hashes, source identity, hardware/driver information, environment,
provenance, and memory counters. A failed correctness check or idle guard is
written to the JSON report before the process exits with an error. Preserve
that report and use a new path for a subsequent attempt.

Run the CPU-only validation self-test with:

```text
Rscript tools/test-matmul-precision-validation.R
```
