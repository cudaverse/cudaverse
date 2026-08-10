if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking benchmark reports requires jsonlite.", call. = FALSE)
}

input <- Sys.getenv("CUDAVERSE_BENCHMARK_REPORT", unset = "")
if (!nzchar(input) || !file.exists(input)) {
  stop("Set CUDAVERSE_BENCHMARK_REPORT to an existing report.", call. = FALSE)
}
report <- jsonlite::read_json(input, simplifyVector = FALSE)
scalar <- function(x) unlist(x, recursive = TRUE, use.names = FALSE)[[1L]]
number <- function(x) as.numeric(scalar(x))
logical_value <- function(x) isTRUE(as.logical(scalar(x)))
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

failures <- character()
require_gate <- function(value, message) {
  if (!isTRUE(value)) failures <<- c(failures, message)
}

require_gate(
  identical(scalar(report$schema), "cudaverse-benchmark/1"),
  "unexpected benchmark report schema"
)
profile <- scalar(report$profile)
require_gate(profile %in% c("smoke", "full"), "invalid report profile")
require_gate(logical_value(report$complete), "report is incomplete")
require_gate(
  nchar(scalar(report$source$commit)) == 40L,
  "report does not identify one source commit"
)
if (identical(profile, "full")) {
  require_gate(
    !logical_value(report$source$tracked_dirty),
    "full report source contains tracked changes"
  )
}
require_gate(
  number(report$installed_size_bytes$cudaverse) > 0 &&
    number(report$installed_size_bytes$bundled_cuda_runtime) == 0,
  "installed-size or bundled-runtime contract failed"
)

contract_path <- Sys.getenv(
  "CUDAVERSE_BENCHMARK_CONTRACT",
  unset = file.path("inst", "benchmarks", "contract.csv")
)
contract <- utils::read.csv(
  contract_path, stringsAsFactors = FALSE, check.names = FALSE,
  na.strings = c("", "NA")
)
expected <- contract$case_id[contract$profile == profile]
require_gate(
  setequal(names(report$cases), expected),
  "report cases do not match the selected contract profile"
)
required_backends <- unlist(
  report$contract$backends, recursive = TRUE, use.names = FALSE
)
require_gate(
  identical(required_backends[[1L]], "base"),
  "base backend did not establish references first"
)
if (!is.null(report$contract$stage_sampling)) {
  require_gate(
    identical(
      scalar(report$contract$stage_sampling),
      paste(
        "pipeline stages are collected from the same synchronized timed",
        "host-boundary runs"
      )
    ),
    "report has an unknown stage-sampling contract"
  )
}
if (!is.null(report$contract$memory_sampling)) {
  require_gate(
    identical(
      scalar(report$contract$memory_sampling),
      paste(
        "one separate instrumented execution after timing; allocator tracking",
        "is excluded from retained timing samples"
      )
    ),
    "report has an unknown memory-sampling contract"
  )
}

for (case_id in expected) {
  case <- report$cases[[case_id]]
  require_gate(!is.null(case), paste(case_id, "is missing"))
  if (is.null(case)) next
  require_gate(
    setequal(names(case$backends), required_backends),
    paste(case_id, "does not contain every requested backend")
  )
  for (backend in required_backends) {
    value <- case$backends[[backend]]
    label <- paste(case_id, backend)
    require_gate(
      identical(scalar(value$status), "complete"),
      paste(label, "did not complete")
    )
    require_gate(
      logical_value(value$validation$passed),
      paste(label, "failed numerical parity")
    )
    require_gate(
      number(value$cold_seconds$host_boundary) >= 0 &&
        number(value$warm$host_boundary$median_seconds) >= 0 &&
        number(value$warm$host_boundary$p95_seconds) >= 0,
      paste(label, "has invalid timing values")
    )
    runs <- unlist(
      value$warm$host_boundary$runs_seconds,
      recursive = TRUE, use.names = FALSE
    )
    definition <- contract[contract$case_id == case_id, , drop = FALSE]
    require_gate(
      length(runs) == definition$timed_runs,
      paste(label, "has the wrong timed-run count")
    )
    require_gate(
      identical(scalar(value$provenance$pca$schema %||%
                         value$provenance$schema), "cudaverse-stage/1"),
      paste(label, "does not contain cudaverse-stage/1 provenance")
    )
    require_gate(
      !is.null(value$memory$backend_allocator_peak_source),
      paste(label, "does not document peak-memory provenance")
    )
  }
}

if (length(failures)) {
  stop(
    "Benchmark report failed:\n- ",
    paste(failures, collapse = "\n- "),
    call. = FALSE
  )
}
message("Benchmark report passed all machine-readable gates: ", input)
