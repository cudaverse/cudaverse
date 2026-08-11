sys.source(file.path("tools", "benchmark-timing.R"), envir = environment())

calls <- 0L
clock_value <- 0
clock <- function() {
  clock_value <<- clock_value + 0.25
  clock_value
}
run <- function() {
  calls <<- calls + 1L
  list(call = calls, seconds = c(stage_a = calls, stage_b = calls + 0.5))
}
summarize <- function(values) list(runs_seconds = values)

result <- benchmark_time_runs(
  cold_run = run,
  timed_run = run,
  warmups = 2L,
  timed_runs = 3L,
  summarize = summarize,
  collect = function(value) value$seconds,
  clock = clock
)

stopifnot(
  identical(calls, 6L),
  identical(result$cold_seconds, 0.25),
  identical(result$warm$runs_seconds, rep(0.25, 3L)),
  identical(length(result$observations), 3L),
  identical(vapply(result$observations, `[[`, numeric(1L), "stage_a"),
            c(4, 5, 6)),
  identical(result$last$call, 6L)
)

without_collection <- benchmark_time_runs(
  cold_run = function() TRUE,
  timed_run = function() TRUE,
  warmups = 0L,
  timed_runs = 1L,
  summarize = summarize,
  clock = clock
)
stopifnot(is.null(without_collection$observations))

message("Benchmark timing/collection self-tests passed.")
