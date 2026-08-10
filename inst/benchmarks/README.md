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
