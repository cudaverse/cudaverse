if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Benchmark report self-tests require jsonlite.", call. = FALSE)
}

run_checker <- function(path) {
  old <- Sys.getenv("CUDAVERSE_BENCHMARK_REPORT", unset = NA_character_)
  on.exit({
    Sys.unsetenv("CUDAVERSE_BENCHMARK_REPORT")
    if (!is.na(old)) Sys.setenv(CUDAVERSE_BENCHMARK_REPORT = old)
  }, add = TRUE)
  Sys.setenv(CUDAVERSE_BENCHMARK_REPORT = path)
  sys.source(
    file.path("tools", "check-benchmark-report.R"),
    envir = new.env(parent = globalenv())
  )
}
expect_failure <- function(code, pattern) {
  error <- tryCatch({ force(code); NULL }, error = identity)
  if (!inherits(error, "error") ||
      !grepl(pattern, conditionMessage(error), fixed = TRUE)) {
    stop("Expected an error containing: ", pattern, call. = FALSE)
  }
}
contract <- utils::read.csv(
  file.path("inst", "benchmarks", "contract.csv"),
  stringsAsFactors = FALSE, check.names = FALSE,
  na.strings = c("", "NA")
)
contract_list <- lapply(seq_len(nrow(contract)), function(index) {
  value <- as.list(contract[index, , drop = FALSE])
  lapply(value, function(field) if (is.na(field[[1L]])) NULL else field[[1L]])
})
full_contract_list <- contract_list[contract$profile == "full"]
summary_value <- function(runs = rep(1, 10L)) list(
  median_seconds = unname(stats::median(runs)),
  p95_seconds = unname(stats::quantile(runs, 0.95, type = 8)),
  runs_seconds = as.list(runs)
)
backend_value <- function(definition, backend) {
  resident_field <- if (identical(definition$family, "matmul")) {
    "resident_compute"
  } else if (identical(definition$family, "sparse_pca_knn")) {
    "resident_continuation"
  } else {
    NULL
  }
  cold <- list(host_boundary = 1)
  warm <- list(host_boundary = summary_value())
  if (is.null(resident_field)) {
    cold$resident_continuation <- NULL
    warm$resident_continuation <- list(
      status = "not_separable", reason = "synthetic"
    )
  } else {
    cold[[resident_field]] <- 1
    warm[[resident_field]] <- summary_value()
  }
  if (!identical(definition$family, "matmul")) {
    warm$stages <- setNames(
      replicate(5L, summary_value(), simplify = FALSE),
      c("explicit_transfer", "normalization", "pca", "knn",
        "full_pipeline")
    )
  }
  provenance <- if (identical(definition$family, "matmul")) {
    list(schema = "cudaverse-stage/1")
  } else {
    list(pca = list(schema = "cudaverse-stage/1"))
  }
  list(
    status = "complete", cold_seconds = cold, warm = warm,
    validation = list(passed = TRUE), provenance = provenance,
    memory = list(
      backend_allocator_peak_bytes = 0,
      backend_allocator_peak_source = "synthetic",
      tracked_current_post_cleanup_difference_bytes = if (
        identical(backend, "native")
      ) 0 else NULL
    )
  )
}
cases <- list()
for (definition in full_contract_list) {
  cases[[definition$case_id]] <- list(
    definition = definition,
    backends = setNames(lapply(
      c("base", "native", "torch"),
      function(backend) backend_value(definition, backend)
    ), c("base", "native", "torch"))
  )
}
report <- list(
  schema = "cudaverse-benchmark/1", profile = "full",
  source = list(
    commit = paste(rep("a", 40L), collapse = ""), tracked_dirty = FALSE
  ),
  installed_size_bytes = list(
    cudaverse = 1, bundled_cuda_runtime = 0
  ),
  contract = list(
    backends = as.list(c("base", "native", "torch")),
    cases = full_contract_list
  ),
  cases = cases, complete = TRUE
)
work <- tempfile("cudaverse-benchmark-report-")
dir.create(work)
path <- file.path(work, "report.json")
write_report <- function(value = report) jsonlite::write_json(
  value, path, auto_unbox = TRUE, pretty = TRUE, null = "null",
  digits = 16
)
write_report()
run_checker(path)

changed <- report
changed$contract$backends <- as.list(c("base", "native"))
for (name in names(changed$cases)) changed$cases[[name]]$backends$torch <- NULL
write_report(changed)
expect_failure(run_checker(path), "full report must contain base, native, and torch")

changed <- report
changed$cases[[1L]]$definition$rows <- 999
write_report(changed)
expect_failure(run_checker(path), "definition does not match contract.csv")

changed <- report
changed$cases[[1L]]$backends$native$warm$host_boundary$runs_seconds[[1L]] <- -1
write_report(changed)
expect_failure(run_checker(path), "finite non-negative timed runs")

changed <- report
changed$cases[[1L]]$backends$native$warm$host_boundary$median_seconds <- 2
write_report(changed)
expect_failure(run_checker(path), "summary does not match its retained runs")

changed <- report
changed$cases[[1L]]$backends$native$memory$backend_allocator_peak_bytes <- -1
write_report(changed)
expect_failure(run_checker(path), "invalid peak-memory evidence")

changed <- report
changed$cases[[1L]]$backends$native$memory[[
  "tracked_current_post_cleanup_difference_bytes"
]] <- NULL
write_report(changed)
expect_failure(run_checker(path), "retains tracked native bytes")

message("Benchmark report positive and rejection self-tests passed.")
