sys.source(
  file.path("tools", "benchmark-gpu-guard.R"),
  envir = environment()
)
stopifnot(isTRUE(benchmark_validate_idle_gpu_requirement(
  "full", c("base", "native"), TRUE
)))
stopifnot(isTRUE(benchmark_validate_idle_gpu_requirement(
  "smoke", c("base", "native"), FALSE
)))
error <- tryCatch(benchmark_validate_idle_gpu_requirement(
  "full", c("base", "native"), FALSE
), error = identity)
stopifnot(inherits(error, "error"),
          grepl("require idle-GPU inspection", conditionMessage(error),
                fixed = TRUE))

stopifnot(identical(
  benchmark_parse_compute_pids(c(" 42 ", "17", "42")),
  c(17L, 42L)
))
stopifnot(identical(
  benchmark_parse_compute_pids(c("No running processes found", "")),
  integer()
))
stopifnot(identical(benchmark_parse_compute_pids(character()), integer()))
for (invalid in list("N/A", "permission denied", "42 extra", "42.5", "0",
                     "-1", "2147483648", NA_character_,
                     c("42", "unrecognized"),
                     c("42", "No running processes found"))) {
  error <- tryCatch(benchmark_parse_compute_pids(invalid), error = identity)
  stopifnot(inherits(error, "error"),
            grepl("refusing to assume", conditionMessage(error), fixed = TRUE))
}
for (invalid in list(NA_integer_, 42.5, -1L, "42", Inf)) {
  error <- tryCatch(
    benchmark_assert_idle_gpu("self-test", invalid, current_pid = 42L),
    error = identity
  )
  stopifnot(inherits(error, "error"))
}
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
