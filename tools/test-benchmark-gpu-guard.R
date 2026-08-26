sys.source(
  file.path("tools", "benchmark-gpu-guard.R"),
  envir = environment()
)

stopifnot(identical(
  benchmark_parse_compute_pids(c(" 42 ", "17", "42")),
  c(17L, 42L)
))
stopifnot(identical(
  benchmark_parse_compute_pids(c("No running processes found", "")),
  integer()
))
stopifnot(isTRUE(benchmark_assert_idle_gpu(
  "self-test", pids = c(42L), current_pid = 42L
)))

error <- tryCatch(
  benchmark_assert_idle_gpu(
    "self-test", pids = c(42L, 99L), current_pid = 42L
  ),
  error = identity
)
stopifnot(
  inherits(error, "error"),
  grepl("competing compute process", conditionMessage(error), fixed = TRUE),
  grepl("99", conditionMessage(error), fixed = TRUE)
)

message("Benchmark GPU guard positive and rejection self-tests passed.")
