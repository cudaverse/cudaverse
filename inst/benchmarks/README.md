# cudaverse benchmark contract

`contract.csv` is the authoritative workload definition established for the
0.3 candidate and retained as the 0.4 regression baseline until a versioned
replacement is justified. It has two profiles:

- `smoke` proves the runner, report schema, numerical comparisons, provenance,
  timing distributions, and memory fields on bounded inputs;
- `full` contains float32 and float64 square matrix multiplication at 256,
  1024, and 4096, plus dense and sparse PCA/kNN at 1,000 x 50,
  10,000 x 100, and 50,000 x 128 with `k = 15`.

Every full case uses five warmups followed by ten timed runs. The runner records
the first workload time, warm median/p95 and raw observations, synchronized
stage times, host-boundary and resident timings where the public API makes that
separation possible, backend allocator and whole-device memory observations,
installed size, numerical error, and `cudaverse-stage/1` provenance.
Memory observations use the same public `cuda_memory_info()` contract shown to
users; native peak reset remains a measurement-only maintainer action outside
retained timing samples.
Pipeline stage observations are captured from the same ten timed host-boundary
runs rather than executing a second ten-run stage pass. Sparse resident timing
still has its own five warmups and ten timed runs because it measures a distinct
preloaded-input boundary. Memory measurement remains a separate instrumented
execution so allocator tracking cannot perturb the retained timing samples.

Dense PCA currently accepts host data at its public boundary. Its internal
backend upload is therefore included in the PCA stage and is explicitly marked
as not separately measurable. Sparse upload and matmul tensor upload are
separate public operations and receive both host-boundary and resident timing.
This distinction prevents subtraction-based or inferred transfer numbers from
being presented as measurements.

Run a protected-machine smoke report from a clean, installed source commit:

```powershell
$env:CUDAVERSE_BENCHMARK_PROFILE = "smoke"
$env:CUDAVERSE_BENCHMARK_BACKENDS = "base,native,torch"
$env:NVIDIA_TF32_OVERRIDE = "0" # Required when the standard torch backend runs
$env:CUDAVERSE_BENCHMARK_OUTPUT = "C:/evidence/benchmark-smoke.json"
Rscript tools/run-benchmark-contract.R

$env:CUDAVERSE_BENCHMARK_REPORT = "C:/evidence/benchmark-smoke.json"
Rscript tools/check-benchmark-report.R
```

Use `base,native` if torch is not part of the comparison. The standard benchmark
never enables the new per-call TF32 mode. Compare that mode separately with
`tools/run-matmul-precision-experiment.R`; its usage and assumptions are in
`tools/README-matmul-precision.md`.

Before a retained `full` run, create an exact installation from the clean
checkout. The destination must be new and outside the checkout:

```powershell
Rscript tools/install-benchmark-candidate.R C:/evidence/candidate
$env:R_LIBS_USER = "C:/evidence/candidate/lib;$env:R_LIBS_USER"
$env:CUDAVERSE_BENCHMARK_INSTALL_MANIFEST = "C:/evidence/candidate/manifest.json"
$env:CUDAVERSE_BENCHMARK_PROFILE = "full"
$env:CUDAVERSE_BENCHMARK_OUTPUT = "C:/evidence/benchmark-full.json"
Rscript tools/run-benchmark-contract.R
```

Keep the dependency library containing `jsonlite`, `digest` and `Matrix`
available. The installer archives the exact commit, performs a clean compile
and records every installed file's SHA-256, including native code and R
bytecode. Full runs verify source and installed files before and after the
workload. Matching package versions alone is insufficient. Put outputs outside
the checkout so they do not dirty the source. Dirty full runs are rejected even
when `CUDAVERSE_BENCHMARK_ALLOW_DIRTY=true` is set for exploratory smoke work.
The report is raw machine evidence,
not a universal speed claim; workload-specific interpretation belongs in the
candidate benchmark assessment.

A full run that includes a CUDA backend also requires the GPU to have no
competing compute process. The runner checks at startup and around every
backend measurement, excluding its own R process. If another workload appears,
the run stops before retaining that measurement and leaves the last atomic
checkpoint available for review. Set
`CUDAVERSE_BENCHMARK_REQUIRE_IDLE_GPU=false` is allowed only for exploratory
`smoke` measurements. A full CUDA run cannot disable the guard. The guard samples
compute activity and does not provide continuous monitoring or establish that
unreported graphics activity is absent; use a controlled machine window.

## Standard float32 numerical policy

`standard-fp32-dot-product/1` uses the componentwise absolute-product sum
`S = |A| |B|` and contracted dimension `k`. For unit roundoff `u`, define
`gamma(n,u) = n*u/(1-n*u)`. The double-computed sum is inflated to an upper
bound `S_upper`; acceptance allows
`(gamma(2k,2^-24) + gamma(2k,2^-53))*S_upper + U`, with a small documented
binary64 bound-arithmetic inflation and
`U = 2k*2^-126/(1-2k*2^-24)` for underflow under the stated operation model.
Zero absolute-product support must produce exact zeros. Nonfinite values,
subnormal inputs, empty shapes and possible float32 overflow are rejected.

This model accounts for cancellation through each output's own `S`. It is an
explicit benchmark contract under ordinary full-mantissa FP32 operations,
not a universal bound for every undocumented cuBLAS algorithm. The original
`1e-6 + 1e-5*abs(reference)` comparison remains a separate diagnostic, including
its failed-entry count. Reports and checkpoints identify the policy version
and a line-ending-independent validator fingerprint. Application accuracy
requirements must still be assessed for the complete analysis.

The runner logs `started` and `complete` events for cold, every warmup, every
timed run, and the separate memory pass within each case/backend/scope. Progress
callbacks run outside retained timing intervals, so the logged completion
duration does not include logging or garbage collection. These messages make a
multi-hour backend observable but are not checkpoints or evidence: only a
fully validated backend written to the JSON can be resumed or counted.

After the complete report passes its machine-readable gate, generate and check
the human-readable assessment from that exact file:

```powershell
$env:CUDAVERSE_BENCHMARK_REPORT = "benchmark-full.json"
$env:CUDAVERSE_BENCHMARK_SUMMARY = "benchmark-full.md"
Rscript tools/summarize-benchmark-report.R
Rscript tools/check-benchmark-summary.R
```

The summary records the report SHA-256 and source commit, and contains every
case/backend timing, validation, footprint, and peak-memory row. The checker
rejects an incomplete report, a draft summary, a mismatched digest, or a
missing case/backend row. Automatically generated ratios are descriptive
comparisons of ten-run sample medians, not confidence intervals or statistical
significance tests. During a long protected-machine run, setting
`CUDAVERSE_BENCHMARK_ALLOW_INCOMPLETE=true` can produce a visibly marked draft
from recoverably checkpointed results. Such a draft is for monitoring only and
cannot pass the retained-summary checker.

Each backend completion is first written and parsed as a staging JSON. The
runner then rotates the last valid report to `<output>.previous` before
installing the new checkpoint. A completed report removes that recovery file
only after its `complete = true` JSON has been parsed successfully. If an
interrupted filesystem write leaves the current path invalid, recover the last
fully parsed checkpoint without treating it as final evidence:

```r
sys.source("tools/benchmark-checkpoint-io.R", envir = environment())
recover_benchmark_checkpoint("benchmark-full.json")
```

Recovery preserves completed machine evidence but does not invent unrecorded
timings or mark an incomplete run complete. The final report and summary still
have to pass their normal checkers.

To continue an interrupted run, ask the runner to validate and resume the same
output explicitly:

```powershell
$env:CUDAVERSE_BENCHMARK_RESUME = "true"
Rscript tools/run-benchmark-contract.R
```

Resume is intentionally strict. It requires the same clean source commit,
package/R/torch versions, GPU identity, profile, backend order, and ordered case
contract. Only a case with every requested backend marked complete and passing
validation is reused. An incomplete case is discarded and rerun as a whole,
starting with the base reference, so a partial backend result cannot be joined
to a missing or different reference.

By default, a non-resume run refuses to replace an existing output or recovery
file. Set `CUDAVERSE_BENCHMARK_OVERWRITE=true` only when intentionally starting
fresh. Resume and overwrite cannot be enabled together.
