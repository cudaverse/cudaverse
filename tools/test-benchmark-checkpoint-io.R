if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Benchmark checkpoint self-tests require jsonlite.", call. = FALSE)
}
sys.source(
  file.path("tools", "benchmark-checkpoint-io.R"),
  envir = environment()
)

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

work <- tempfile("cudaverse-benchmark-checkpoint-")
dir.create(work)
path <- file.path(work, "report.json")
previous <- benchmark_checkpoint_previous(path)

first <- list(schema = "cudaverse-benchmark/1", sequence = 1L,
              complete = FALSE)
second <- list(schema = "cudaverse-benchmark/1", sequence = 2L,
               complete = FALSE)
final <- list(schema = "cudaverse-benchmark/1", sequence = 3L,
              complete = TRUE)

write_benchmark_checkpoint(first, path)
stopifnot(benchmark_checkpoint_valid(path), !file.exists(previous))
write_benchmark_checkpoint(second, path)
stopifnot(
  identical(jsonlite::read_json(path)$sequence, 2L),
  identical(jsonlite::read_json(previous)$sequence, 1L)
)

writeLines("interrupted write", path, useBytes = TRUE)
stopifnot(identical(recover_benchmark_checkpoint(path), "previous"))
stopifnot(
  identical(jsonlite::read_json(path)$sequence, 1L),
  !file.exists(previous)
)

expect_error_message(
  finalize_benchmark_checkpoint(path),
  "Cannot finalize an incomplete"
)
write_benchmark_checkpoint(final, path)
stopifnot(file.exists(previous))
finalize_benchmark_checkpoint(path)
stopifnot(
  benchmark_checkpoint_valid(path),
  identical(jsonlite::read_json(path)$sequence, 3L),
  !file.exists(previous)
)
jsonlite::write_json(
  list(schema = "not-a-benchmark", complete = TRUE),
  path, auto_unbox = TRUE
)
stopifnot(!benchmark_checkpoint_valid(path))

backend_result <- function(passed = TRUE, status = "complete") {
  list(status = status, validation = list(passed = passed))
}
complete_case <- function() {
  list(
    case_id = "case-a",
    backends = list(
      base = backend_result(),
      native = backend_result(),
      torch = backend_result()
    )
  )
}
expected <- list(
  schema = "cudaverse-benchmark/1",
  profile = "full",
  source = list(commit = "abc123", tracked_dirty = FALSE),
  hardware = list(nvidia_smi = c("GPU A, UUID-A")),
  software = list(R = "R 4.6.0", cudaverse = "0.3.0.9000",
                  torch = NA_character_,
                  installed_source_identity = list(
                    verified = TRUE,
                    source_commit = "abc123", source_tree = "tree-a",
                    manifest_sha256 = "manifest-a",
                    installed_payload_sha256 = "payload-a"
                  )),
  contract = list(
    backends = c("base", "native", "torch"),
    NVIDIA_TF32_OVERRIDE = "0",
    idle_gpu_guard = list(
      required = TRUE,
      sampling = "before and after each sample, outside its timed boundary",
      continuous_monitoring = FALSE
    ),
    numeric_policy = list(version = "standard-fp32-dot-product/1",
                          validator_sha256 = "validator-a"),
    cases = data.frame(
      case_id = c("case-a", "case-b"),
      rows = c(100L, 200L),
      dtype = c("float32", NA_character_)
    )
  )
)
existing <- expected
existing$software$torch <- NULL
existing$contract$cases <- list(
  list(case_id = "case-a", rows = 100, dtype = "float32"),
  list(case_id = "case-b", rows = 200, dtype = NULL)
)
existing$cases <- list(
  `case-a` = complete_case(),
  `case-b` = complete_case()
)

stopifnot(
  isTRUE(validate_benchmark_resume(existing, expected)),
  benchmark_checkpoint_case_complete(
    existing$cases$`case-a`, expected$contract$backends
  )
)

incomplete <- existing$cases$`case-a`
incomplete$backends$torch <- NULL
stopifnot(!benchmark_checkpoint_case_complete(
  incomplete, expected$contract$backends
))
incomplete <- existing$cases$`case-a`
incomplete$backends$native$validation$passed <- FALSE
stopifnot(!benchmark_checkpoint_case_complete(
  incomplete, expected$contract$backends
))
incomplete <- existing$cases$`case-a`
incomplete$backends$base$status <- "running"
stopifnot(!benchmark_checkpoint_case_complete(
  incomplete, expected$contract$backends
))

# A failing backend must survive the parity error on disk, without discarding
# earlier successful backends or allowing resume to reuse the failed case.
failure_path <- file.path(work, "failed-parity.json")
failure_report <- existing
failure_report$complete <- TRUE
write_benchmark_checkpoint(failure_report, failure_path)
failed_result <- list(
  status = "complete",
  validation = list(passed = FALSE, max_absolute_error = 2e-6,
                    max_tolerance_ratio = 2, failed_elements = 1L,
                    rtol = 1e-5, atol = 1e-6),
  warm = list(host_boundary = list(runs_seconds = c(1, 2))),
  provenance = list(schema = "cudaverse-stage/1"),
  reference = matrix(1, 2L, 2L)
)
expect_error_message({
  checkpoint_benchmark_parity_failure(
    failure_report, "case-a", "native", failed_result, failure_path
  )
  stop("case-a failed parity on backend native.")
}, "case-a failed parity on backend native.")
retained <- jsonlite::read_json(failure_path, simplifyVector = FALSE)
native_failure <- retained$cases$`case-a`$backends$native
stopifnot(
  benchmark_checkpoint_valid(failure_path),
  benchmark_checkpoint_valid(benchmark_checkpoint_previous(failure_path)),
  identical(retained$complete, FALSE),
  identical(native_failure$status, "failed_parity"),
  identical(native_failure$validation$passed, FALSE),
  identical(native_failure$validation$failed_elements, 1L),
  identical(native_failure$validation$max_tolerance_ratio, 2L),
  identical(native_failure$validation$rtol, 1e-5),
  identical(native_failure$validation$atol, 1e-6),
  is.null(native_failure$reference),
  identical(retained$cases$`case-a`$backends$base$status, "complete"),
  !benchmark_checkpoint_case_complete(
    retained$cases$`case-a`, expected$contract$backends
  ),
  benchmark_checkpoint_case_complete(
    retained$cases$`case-b`, expected$contract$backends
  ),
  isTRUE(validate_benchmark_resume(retained, expected))
)
# Even a corrupted TRUE flag cannot make failed_parity reusable.
native_failure$validation$passed <- TRUE
retained$cases$`case-a`$backends$native <- native_failure
stopifnot(!benchmark_checkpoint_case_complete(
  retained$cases$`case-a`, expected$contract$backends
))
expect_error_message(finalize_benchmark_checkpoint(failure_path),
                     "Cannot finalize an incomplete")
expect_error_message(checkpoint_benchmark_parity_failure(
  failure_report, "case-a", "native", backend_result(), failure_path
), "Cannot record a passing result")

expect_resume_rejection <- function(code, pattern) {
  expect_error_message(validate_benchmark_resume(code, expected), pattern)
}
changed <- existing
changed$source$commit <- "other"
expect_resume_rejection(changed, "source commit changed")
changed <- existing
changed$source$tracked_dirty <- TRUE
expect_resume_rejection(changed, "existing report source was dirty")
dirty_expected <- expected
dirty_expected$source$tracked_dirty <- TRUE
expect_error_message(
  validate_benchmark_resume(existing, dirty_expected),
  "current benchmark source is dirty"
)
changed <- existing
changed$profile <- "smoke"
expect_resume_rejection(changed, "benchmark profile changed")
changed <- existing
changed$contract$numeric_policy <- NULL
expect_resume_rejection(changed, "numeric policy version changed")
changed <- existing
changed$contract$numeric_policy$validator_sha256 <- "validator-b"
expect_resume_rejection(changed, "numeric policy fingerprint changed")
changed <- existing
changed$contract$NVIDIA_TF32_OVERRIDE <- "<unset>"
expect_resume_rejection(changed, "TF32 override changed")
changed <- existing
changed$contract$idle_gpu_guard$required <- FALSE
expect_resume_rejection(changed, "idle-GPU guard required changed")
changed <- existing
changed$contract$idle_gpu_guard$required <- c(TRUE, FALSE)
expect_resume_rejection(changed, "idle-GPU guard required changed")
changed <- existing
changed$contract$idle_gpu_guard$sampling <- "start only"
expect_resume_rejection(changed, "idle-GPU guard sampling changed")
changed <- existing
changed$contract$idle_gpu_guard$continuous_monitoring <- TRUE
expect_resume_rejection(changed, "idle-GPU guard continuous_monitoring changed")
changed <- existing
changed$contract$idle_gpu_guard <- NULL
expect_resume_rejection(changed, "idle-GPU guard required changed")
changed <- existing
changed$software$R <- "R 4.6.1"
expect_resume_rejection(changed, "R software identity changed")
changed <- existing
changed$software$installed_source_identity$installed_payload_sha256 <-
  "payload-b"
expect_resume_rejection(changed, "installed package installed_payload_sha256 changed")
changed <- existing
changed$software$installed_source_identity$manifest_sha256 <- "manifest-b"
expect_resume_rejection(changed, "installed package manifest_sha256 changed")
changed <- existing
changed$hardware$nvidia_smi <- "GPU B, UUID-B"
expect_resume_rejection(changed, "GPU identity changed")
changed <- existing
changed$contract$backends <- c("base", "torch", "native")
expect_resume_rejection(changed, "benchmark backend order changed")
changed <- existing
changed$contract$cases <- rev(changed$contract$cases)
expect_resume_rejection(changed, "benchmark case contract changed")
changed <- existing
changed$contract$cases[[1L]]$rows <- 101
expect_resume_rejection(changed, "benchmark case contract changed")

message("Benchmark checkpoint write/recovery/resume self-tests passed.")
