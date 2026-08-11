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
  length(required_backends) > 0L && !anyDuplicated(required_backends) &&
    all(required_backends %in% c("base", "native", "torch")),
  "report contains an invalid backend contract"
)
require_gate(
  identical(required_backends[[1L]], "base"),
  "base backend did not establish references first"
)
if (identical(profile, "full")) {
  require_gate(
    identical(required_backends, c("base", "native", "torch")),
    "full report must contain base, native, and torch in contract order"
  )
}
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

definition_matches <- function(actual, expected) {
  numeric_fields <- c(
    "rows", "columns", "density", "k", "components", "warmups",
    "timed_runs"
  )
  fields <- names(expected)
  all(vapply(fields, function(field) {
    expected_value <- expected[[field]][[1L]]
    actual_value <- actual[[field]]
    if (is.na(expected_value)) return(is.null(actual_value) || !length(actual_value))
    if (field %in% numeric_fields) {
      return(isTRUE(all.equal(
        number(actual_value), as.numeric(expected_value), tolerance = 0
      )))
    }
    identical(as.character(scalar(actual_value)), as.character(expected_value))
  }, logical(1L)))
}

contract_cases <- report$contract$cases
selected_contract <- contract[contract$profile == profile, , drop = FALSE]
require_gate(
  length(contract_cases) == nrow(selected_contract) &&
    identical(
      vapply(contract_cases, function(value) {
        as.character(scalar(value$case_id))
      }, character(1L)),
      selected_contract$case_id
    ),
  "embedded benchmark contract does not match contract.csv case order"
)
if (length(contract_cases) == nrow(selected_contract)) {
  for (index in seq_len(nrow(selected_contract))) {
    require_gate(
      definition_matches(
        contract_cases[[index]], selected_contract[index, , drop = FALSE]
      ),
      paste(
        "embedded benchmark contract differs for",
        selected_contract$case_id[[index]]
      )
    )
  }
}

check_timing_summary <- function(value, expected_runs, label) {
  if (is.null(value)) {
    require_gate(FALSE, paste(label, "timing summary is missing"))
    return(invisible(FALSE))
  }
  runs <- suppressWarnings(as.numeric(unlist(
    value$runs_seconds, recursive = TRUE, use.names = FALSE
  )))
  median_value <- number(value$median_seconds)
  p95_value <- number(value$p95_seconds)
  require_gate(
    length(runs) == expected_runs && all(is.finite(runs)) && all(runs >= 0),
    paste(label, "does not contain finite non-negative timed runs")
  )
  require_gate(
    is.finite(median_value) && median_value >= 0 &&
      is.finite(p95_value) && p95_value >= median_value,
    paste(label, "has an invalid median or p95")
  )
  if (length(runs) == expected_runs && all(is.finite(runs))) {
    require_gate(
      isTRUE(all.equal(
        median_value, unname(stats::median(runs)), tolerance = 1e-12
      )) && isTRUE(all.equal(
        p95_value,
        unname(stats::quantile(runs, 0.95, type = 8)),
        tolerance = 1e-12
      )),
      paste(label, "summary does not match its retained runs")
    )
  }
  invisible(TRUE)
}

for (case_id in expected) {
  case <- report$cases[[case_id]]
  require_gate(!is.null(case), paste(case_id, "is missing"))
  if (is.null(case)) next
  definition <- contract[contract$case_id == case_id, , drop = FALSE]
  require_gate(
    nrow(definition) == 1L && definition_matches(case$definition, definition),
    paste(case_id, "definition does not match contract.csv")
  )
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
    cold_host <- number(value$cold_seconds$host_boundary)
    require_gate(is.finite(cold_host) && cold_host >= 0,
                 paste(label, "has invalid cold host-boundary timing"))
    check_timing_summary(
      value$warm$host_boundary, definition$timed_runs,
      paste(label, "host-boundary")
    )
    resident_field <- if (identical(definition$family, "matmul")) {
      "resident_compute"
    } else if (identical(definition$family, "sparse_pca_knn")) {
      "resident_continuation"
    } else {
      NULL
    }
    if (!is.null(resident_field)) {
      cold_resident <- number(value$cold_seconds[[resident_field]])
      require_gate(is.finite(cold_resident) && cold_resident >= 0,
                   paste(label, "has invalid cold resident timing"))
      check_timing_summary(
        value$warm[[resident_field]], definition$timed_runs,
        paste(label, resident_field)
      )
    } else {
      require_gate(
        identical(scalar(value$warm$resident_continuation$status),
                  "not_separable"),
        paste(label, "does not preserve the dense transfer boundary")
      )
    }
    if (!identical(definition$family, "matmul")) {
      required_stages <- c(
        "explicit_transfer", "normalization", "pca", "knn",
        "full_pipeline"
      )
      require_gate(
        identical(names(value$warm$stages), required_stages),
        paste(label, "does not contain the required pipeline stages")
      )
      for (stage in required_stages) {
        check_timing_summary(
          value$warm$stages[[stage]], definition$timed_runs,
          paste(label, "stage", stage)
        )
      }
    }
    require_gate(
      identical(scalar(value$provenance$pca$schema %||%
                         value$provenance$schema), "cudaverse-stage/1"),
      paste(label, "does not contain cudaverse-stage/1 provenance")
    )
    require_gate(
      !is.null(value$memory$backend_allocator_peak_source),
      paste(label, "does not document peak-memory provenance")
    )
    peak <- number(value$memory$backend_allocator_peak_bytes)
    require_gate(is.finite(peak) && peak >= 0,
                 paste(label, "has invalid peak-memory evidence"))
    tracked <- value$memory$tracked_current_post_cleanup_difference_bytes
    tracked_values <- unlist(tracked, recursive = TRUE, use.names = FALSE)
    require_gate(
      if (identical(backend, "native")) {
        length(tracked_values) == 1L &&
          identical(suppressWarnings(as.numeric(tracked_values[[1L]])), 0)
      } else {
        !length(tracked_values) ||
          identical(suppressWarnings(as.numeric(tracked_values[[1L]])), 0)
      },
      paste(label, "retains tracked native bytes after cleanup")
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
