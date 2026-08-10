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

message("Benchmark checkpoint write/recovery self-tests passed.")
