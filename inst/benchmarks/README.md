# cudaverse 0.3 benchmark contract

`contract.csv` is the authoritative workload definition for 0.3 benchmark
evidence. It has two profiles:

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
$env:CUDAVERSE_BENCHMARK_OUTPUT = "benchmark-smoke.json"
Rscript tools/run-benchmark-contract.R

$env:CUDAVERSE_BENCHMARK_REPORT = "benchmark-smoke.json"
Rscript tools/check-benchmark-report.R
```

Change the profile to `full` only for the retained candidate run. A full report
from a dirty source tree fails validation. The report is raw machine evidence,
not a universal speed claim; workload-specific interpretation belongs in the
candidate benchmark assessment.

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
missing case/backend row. During a long protected-machine run, setting
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
