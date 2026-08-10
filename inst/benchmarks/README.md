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
from atomically completed results. Such a draft is for monitoring only and
cannot pass the retained-summary checker.
