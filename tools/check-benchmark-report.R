if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Checking benchmark reports requires jsonlite and digest.",
       call. = FALSE)
}

input <- Sys.getenv("CUDAVERSE_BENCHMARK_REPORT", unset = "")
if (!nzchar(input) || !file.exists(input)) {
  stop("Set CUDAVERSE_BENCHMARK_REPORT to an existing report.", call. = FALSE)
}
report <- jsonlite::read_json(input, simplifyVector = FALSE)
scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (length(value) != 1L) default else value[[1L]]
}
number <- function(x) as.numeric(scalar(x))
model_number_matches <- function(value, expected) {
  observed <- suppressWarnings(number(value))
  is.finite(observed) && is.finite(expected) &&
    abs(observed - expected) <= max(1e-300, abs(expected) * 1e-12)
}
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
sys.source(
  file.path("tools", "benchmark-validation.R"), envir = environment()
)
numeric_policy <- benchmark_float32_matmul_policy()
policy_fields_match <- function(observed, expected) {
  if (!is.list(observed) || !is.list(expected) ||
      !setequal(names(observed), names(expected)) ||
      anyDuplicated(names(observed))) return(FALSE)
  all(vapply(names(expected), function(field) {
    observed_value <- observed[[field]]
    expected_value <- expected[[field]]
    if (is.list(expected_value)) {
      return(policy_fields_match(observed_value, expected_value))
    }
    if (is.numeric(expected_value)) {
      return(model_number_matches(observed_value, expected_value))
    }
    identical(scalar(observed_value), expected_value)
  }, logical(1L)))
}
require_gate(
  identical(scalar(report$contract$numeric_policy$version),
            numeric_policy$version),
  "report has an unknown float32 numeric policy version"
)
require_gate(
  identical(
    scalar(report$contract$numeric_policy$validator_sha256),
    benchmark_validation_source_sha256()
  ),
  "report float32 numeric policy fingerprint differs from the validator"
)
require_gate(
  policy_fields_match(
    report$contract$numeric_policy,
    c(numeric_policy,
      list(validator_sha256 = benchmark_validation_source_sha256()))
  ),
  "report float32 numeric policy definition differs from the validator"
)
require_gate(
  nchar(scalar(report$source$commit)) == 40L,
  "report does not identify one source commit"
)
if (identical(profile, "full")) {
  require_gate(
    !logical_value(report$source$tracked_dirty),
    "full report source contains tracked changes"
  )
  identity <- report$software$installed_source_identity
  require_gate(
    is.list(identity) && identical(scalar(identity$verified), TRUE) &&
      is.null(identity$reason) &&
      identical(scalar(identity$source_commit),
                scalar(report$source$commit)) &&
      isTRUE(grepl("^[0-9a-f]{40}$", scalar(identity$source_tree))) &&
      isTRUE(grepl("^[0-9a-f]{64}$", scalar(identity$manifest_sha256))) &&
      isTRUE(grepl("^[0-9a-f]{64}$",
                   scalar(identity$installed_payload_sha256))),
    "full report lacks verified exact-source package installation"
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
  !"torch" %in% required_backends ||
    identical(scalar(report$contract$NVIDIA_TF32_OVERRIDE), "0"),
  "standard torch report did not disable implicit TF32"
)
require_gate(
  length(required_backends) > 0L && !anyDuplicated(required_backends) &&
    all(required_backends %in% c("base", "native", "torch")),
  "report contains an invalid backend contract"
)
guard <- report$contract$idle_gpu_guard
require_gate(
  is.list(guard) &&
    (identical(scalar(guard$required), TRUE) ||
       identical(scalar(guard$required), FALSE)) &&
    identical(scalar(guard$sampling),
              "before and after each sample, outside its timed boundary") &&
    identical(scalar(guard$continuous_monitoring), FALSE),
  "report has an invalid idle-GPU guard contract"
)
require_gate(
  length(required_backends) > 0L &&
    identical(required_backends[[1L]], "base"),
  "base backend did not establish references first"
)
if (identical(profile, "full")) {
  require_gate(
    all(c("base", "native") %in% required_backends),
    "full report must declare both base and native backends"
  )
  require_gate(
    !any(required_backends != "base") ||
      identical(scalar(guard$required), TRUE),
    "full CUDA report did not require idle-GPU inspection"
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
    identical(names(case$backends), required_backends),
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
    if (identical(definition$family, "matmul") &&
        identical(definition$dtype, "float32")) {
      validation <- value$validation
      strict <- validation$strict_diagnostic
      inner <- as.double(definition$columns)
      operations <- benchmark_float32_dot_operation_count(inner,
                                                           numeric_policy)
      expected_gamma32 <- operations * numeric_policy$unit_roundoff_float32 /
        (1 - operations * numeric_policy$unit_roundoff_float32)
      expected_gamma64 <- operations * numeric_policy$unit_roundoff_float64 /
        (1 - operations * numeric_policy$unit_roundoff_float64)
      expected_underflow <- operations * numeric_policy$min_normal_float32 /
        (1 - operations * numeric_policy$unit_roundoff_float32)
      max_products <- number(validation$max_sum_absolute_products)
      max_products_upper <- max_products / (1 - expected_gamma64) *
        numeric_policy$bound_arithmetic_inflation
      max_float32 <- (2 - 2^-23) * 2^127
      expected_max_allowance <- (
        (expected_gamma32 + expected_gamma64) *
          max_products_upper +
          expected_underflow
      ) * numeric_policy$bound_arithmetic_inflation
      require_gate(
        identical(scalar(validation$policy_version), numeric_policy$version) &&
          identical(scalar(validation$comparison),
                    numeric_policy$comparison) &&
          is.null(validation$unsupported_reason) &&
          number(validation$inner_dimension) == inner &&
          model_number_matches(validation$unit_roundoff_float32,
                               numeric_policy$unit_roundoff_float32) &&
          model_number_matches(validation$unit_roundoff_float64,
                               numeric_policy$unit_roundoff_float64) &&
          model_number_matches(validation$gamma_float32,
                               expected_gamma32) &&
          model_number_matches(validation$gamma_reference_float64,
                               expected_gamma64) &&
          model_number_matches(validation$underflow_absolute_bound,
                               expected_underflow) &&
          model_number_matches(validation$bound_arithmetic_inflation,
                               numeric_policy$bound_arithmetic_inflation) &&
          is.finite(number(validation$max_sum_absolute_products)) &&
          number(validation$max_sum_absolute_products) >= 0 &&
          is.finite(max_products_upper) &&
          (1 + expected_gamma32) * max_products_upper +
            expected_underflow <= max_float32 &&
          is.finite(number(validation$max_allowance)) &&
          number(validation$max_allowance) >= 0 &&
          model_number_matches(validation$max_allowance,
                               expected_max_allowance) &&
          is.finite(number(validation$max_absolute_error)) &&
          number(validation$max_absolute_error) >= 0 &&
          number(validation$max_absolute_error) <=
            number(validation$max_allowance) * (1 + 1e-12) &&
          is.finite(number(validation$max_relative_error)) &&
          number(validation$max_relative_error) >= 0 &&
          is.finite(number(validation$max_scaled_error)) &&
          number(validation$max_scaled_error) >= 0 &&
          number(validation$max_scaled_error) <= 1 + 32 * 2^-53 &&
          number(validation$max_absolute_error) <=
            number(validation$max_scaled_error) *
            number(validation$max_allowance) * (1 + 1e-12) + 1e-300 &&
          number(validation$failed_elements) == 0,
        paste(label, "lacks a passing standard float32 dot-product bound")
      )
      require_gate(
        !is.null(strict) &&
          identical(
            scalar(strict$comparison),
            "elementwise abs(actual-reference) <= atol+rtol*abs(reference)"
          ) &&
          isTRUE(all.equal(number(strict$rtol), 1e-5, tolerance = 0)) &&
          isTRUE(all.equal(number(strict$atol), 1e-6, tolerance = 0)) &&
          logical_value(strict$shape_matches) && logical_value(strict$finite) &&
          is.finite(number(strict$max_absolute_error)) &&
          number(strict$max_absolute_error) >= 0 &&
          model_number_matches(validation$max_absolute_error,
                               number(strict$max_absolute_error)) &&
          is.finite(number(strict$max_relative_error)) &&
          number(strict$max_relative_error) >= 0 &&
          model_number_matches(validation$max_relative_error,
                               number(strict$max_relative_error)) &&
          is.finite(number(strict$max_tolerance_ratio)) &&
          number(strict$max_tolerance_ratio) >= 0 &&
          is.finite(number(strict$failed_elements)) &&
          number(strict$failed_elements) >= 0 &&
          number(strict$failed_elements) == floor(number(strict$failed_elements)) &&
          identical(logical_value(strict$passed),
                    number(strict$failed_elements) == 0),
        paste(label, "omits the original strict float32 diagnostic")
      )
    }
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
    if (identical(definition$family, "matmul")) {
      stages <- value$provenance$stages
      standard <- is.list(stages) && any(vapply(stages, function(stage) {
        identical(scalar(stage$stage), "matrix_multiply") &&
          identical(scalar(stage$selection_reason), "matmul_standard")
      }, logical(1L)))
      require_gate(standard,
                   paste(label, "is not standard-precision matmul provenance"))
    }
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
