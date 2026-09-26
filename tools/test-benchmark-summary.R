if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop(
    "Benchmark-summary self-tests require jsonlite and digest.",
    call. = FALSE
  )
}

run_script <- function(path) {
  sys.source(path, envir = new.env(parent = globalenv()))
}
expect_error_message <- function(code, pattern) {
  error <- tryCatch({
    force(code)
    NULL
  }, error = identity)
  if (!inherits(error, "error") ||
      !grepl(pattern, conditionMessage(error), fixed = TRUE)) {
    stop("Expected an error containing: ", pattern, call. = FALSE)
  }
}
timing <- function(value) {
  list(
    median_seconds = value,
    p95_seconds = value * 1.1,
    runs_seconds = as.list(rep(value, 10L))
  )
}
memory <- function(bytes) {
  list(
    backend_allocator_peak_bytes = bytes,
    backend_allocator_peak_source = "synthetic allocator",
    tracked_current_post_cleanup_difference_bytes = 0,
    whole_device_used_before_bytes = 0,
    whole_device_used_with_result_bytes = bytes,
    whole_device_used_after_cleanup_bytes = 0,
    whole_device_post_cleanup_absolute_difference_bytes = 0
  )
}
matmul_backend <- function(value, error = 0, strict_failed = 0L) {
  list(
    status = "complete",
    cold_seconds = list(host_boundary = value * 1.2),
    warm = list(
      host_boundary = timing(value),
      resident_compute = timing(value / 2)
    ),
    validation = list(
      max_relative_error = error, max_scaled_error = 0.2,
      strict_diagnostic = list(failed_elements = strict_failed),
      passed = TRUE
    ),
    provenance = list(schema = "cudaverse-stage/1"),
    memory = memory(round(value * 1024^2))
  )
}
pipeline_backend <- function(value, self = FALSE) {
  validation <- if (self) {
    list(passed = TRUE, reference_backend = "self")
  } else {
    list(
      pca_projector_max_absolute_error = 1e-12,
      pca_reconstruction_max_relative_error = 1e-12,
      knn_indices_identical = TRUE,
      knn_distance_max_relative_error = 1e-12,
      passed = TRUE
    )
  }
  list(
    status = "complete",
    cold_seconds = list(host_boundary = value * 1.2),
    warm = list(
      host_boundary = timing(value),
      resident_continuation = list(status = "not_separable"),
      stages = list(pca = timing(value / 3), knn = timing(value * 2 / 3))
    ),
    validation = validation,
    provenance = list(pca = list(schema = "cudaverse-stage/1")),
    memory = memory(round(value * 1024^2))
  )
}

work <- tempfile("cudaverse-benchmark-summary-")
dir.create(work)
report_path <- file.path(work, "report.json")
summary_path <- file.path(work, "summary.md")
report <- list(
  schema = "cudaverse-benchmark/1",
  generated_at_utc = "2026-01-01 00:00:00 UTC",
  profile = "full",
  source = list(commit = paste(rep("a", 40L), collapse = ""),
                tracked_dirty = FALSE),
  hardware = list(nvidia_smi = "synthetic GPU"),
  software = list(R = R.version.string, cudaverse = "0.2.0.9000",
                  torch = "synthetic"),
  contract = list(
    backends = list("base", "native", "torch"),
    NVIDIA_TF32_OVERRIDE = "0",
    numeric_policy = list(version = "standard-fp32-dot-product/1")
  ),
  installed_size_bytes = list(
    cudaverse = 1024L, torch = 2048L, bundled_cuda_runtime = 0L
  ),
  cases = list(
    `matmul-test` = list(
      definition = list(family = "matmul", dtype = "float32"),
      backends = list(
        base = matmul_backend(3),
        native = matmul_backend(1, 1e-7, strict_failed = 2L),
        torch = matmul_backend(2, 1e-7)
      )
    ),
    `dense-test` = list(
      definition = list(family = "dense_pca_knn"),
      backends = list(
        base = pipeline_backend(3, self = TRUE),
        native = pipeline_backend(1),
        torch = pipeline_backend(2)
      )
    )
  ),
  complete = TRUE
)
jsonlite::write_json(
  report, report_path, auto_unbox = TRUE, pretty = TRUE, null = "null"
)
Sys.setenv(
  CUDAVERSE_BENCHMARK_REPORT = report_path,
  CUDAVERSE_BENCHMARK_SUMMARY = summary_path
)
Sys.unsetenv("CUDAVERSE_BENCHMARK_ALLOW_INCOMPLETE")
run_script(file.path("tools", "summarize-benchmark-report.R"))
run_script(file.path("tools", "check-benchmark-summary.R"))
for (variant in list(
    list(value = "", label = "<empty>"),
    list(value = "<unset>", label = "<unset>"),
    list(value = "0", label = "0"),
    list(value = NULL, label = "<missing>")
)) {
  report$contract$NVIDIA_TF32_OVERRIDE <- variant$value
  jsonlite::write_json(
    report, report_path, auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
  run_script(file.path("tools", "summarize-benchmark-report.R"))
  run_script(file.path("tools", "check-benchmark-summary.R"))
  override_line <- readLines(summary_path, warn = FALSE)
  override_line <- override_line[startsWith(override_line,
                                            "- NVIDIA_TF32_OVERRIDE:")]
  stopifnot(identical(override_line,
                      paste0("- NVIDIA_TF32_OVERRIDE: `", variant$label, "`")))
}
report$contract$NVIDIA_TF32_OVERRIDE <- ""
jsonlite::write_json(
  report, report_path, auto_unbox = TRUE, pretty = TRUE, null = "null"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))
writeLines(
  sub("NVIDIA_TF32_OVERRIDE: `<empty>`", "NVIDIA_TF32_OVERRIDE: `n/a`",
      readLines(summary_path, warn = FALSE), fixed = TRUE),
  summary_path, useBytes = TRUE
)
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "summary omits the TF32 override"
)
report$contract$NVIDIA_TF32_OVERRIDE <- "0"
jsonlite::write_json(
  report, report_path, auto_unbox = TRUE, pretty = TRUE, null = "null"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))
run_script(file.path("tools", "check-benchmark-summary.R"))
summary_lines <- readLines(summary_path, warn = FALSE)
native_line <- summary_lines[startsWith(summary_lines,
                                        "| matmul-test | native |")]
stopifnot(length(native_line) == 1L,
          grepl("| 2 | 0.2 | pass |", native_line, fixed = TRUE),
          any(grepl("Global-scale relative error", summary_lines,
                    fixed = TRUE)))
if (!identical(summary_lines[[1L]], "# cudaverse full benchmark evidence")) {
  stop("Benchmark summary title does not reflect the report profile.",
       call. = FALSE)
}
if (!any(grepl(
  "synthetic GPU hardware", summary_lines, fixed = TRUE
))) {
  stop("Benchmark summary does not carry the recorded hardware identity.",
       call. = FALSE)
}

writeLines(
  summary_lines[!grepl(
    "Ratios compare ten-run sample medians descriptively.",
    summary_lines, fixed = TRUE
  )],
  summary_path, useBytes = TRUE
)
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "summary overstates descriptive timing ratios"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))
summary_lines <- readLines(summary_path, warn = FALSE)
writeLines(
  sub("| 2 | 0.2 | pass |", "| 0 | 0.2 | pass |",
      summary_lines, fixed = TRUE),
  summary_path, useBytes = TRUE
)
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "does not show its original strict float32 failure count"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))
summary_lines <- readLines(summary_path, warn = FALSE)
writeLines(
  sub("Global-scale relative error", "Max relative error",
      summary_lines, fixed = TRUE),
  summary_path, useBytes = TRUE
)
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "summary mislabels the global-scale relative error statistic"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))
summary_lines <- readLines(summary_path, warn = FALSE)
writeLines(
  summary_lines[!grepl("Original strict failed entries",
                       summary_lines, fixed = TRUE)],
  summary_path, useBytes = TRUE
)
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "summary omits the original strict float32 diagnostic"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))
summary_lines <- readLines(summary_path, warn = FALSE)
writeLines(
  summary_lines[!grepl("Stage sampling:", summary_lines, fixed = TRUE)],
  summary_path, useBytes = TRUE
)
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "summary omits the stage-sampling boundary"
)
run_script(file.path("tools", "summarize-benchmark-report.R"))

report$complete <- FALSE
jsonlite::write_json(
  report, report_path, auto_unbox = TRUE, pretty = TRUE, null = "null"
)
expect_error_message(
  run_script(file.path("tools", "summarize-benchmark-report.R")),
  "The report is incomplete"
)
Sys.setenv(CUDAVERSE_BENCHMARK_ALLOW_INCOMPLETE = "true")
run_script(file.path("tools", "summarize-benchmark-report.R"))
expect_error_message(
  run_script(file.path("tools", "check-benchmark-summary.R")),
  "the benchmark report is incomplete"
)
message("Benchmark-summary positive and rejection self-tests passed.")
