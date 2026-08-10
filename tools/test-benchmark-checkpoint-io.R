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
                  torch = NA_character_),
  contract = list(
    backends = c("base", "native", "torch"),
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
changed$software$R <- "R 4.6.1"
expect_resume_rejection(changed, "R software identity changed")
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
