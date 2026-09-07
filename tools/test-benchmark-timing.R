sys.source(file.path("tools", "benchmark-timing.R"), envir = environment())

progress_messages <- character()
logger <- benchmark_progress_logger(
  "dense-50000x128", "base", "host_boundary",
  reporter = function(value) progress_messages <<- c(progress_messages, value)
)
logger("timed_started", 3L, 10L, NA_real_)
logger("timed_complete", 3L, 10L, 12.5)
stopifnot(
  identical(
    progress_messages,
    c(
      paste0(
        "    progress: dense-50000x128/base/host_boundary ",
        "timed_started 3/10"
      ),
      paste0(
        "    progress: dense-50000x128/base/host_boundary ",
        "timed_complete 3/10 in 12.5 s"
      )
    )
  )
)

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
progress_events <- list()
progress <- function(event, index, total, seconds) {
  progress_events[[length(progress_events) + 1L]] <<- list(
    event = event, index = index, total = total, seconds = seconds
  )
}

result <- benchmark_time_runs(
  cold_run = run,
  timed_run = run,
  warmups = 2L,
  timed_runs = 3L,
  summarize = summarize,
  collect = function(value) value$seconds,
  clock = clock,
  progress = progress
)

stopifnot(
  identical(calls, 6L),
  identical(result$cold_seconds, 0.25),
  identical(result$warm$runs_seconds, rep(0.25, 3L)),
  identical(length(result$observations), 3L),
  identical(vapply(result$observations, `[[`, numeric(1L), "stage_a"),
            c(4, 5, 6)),
  identical(result$last$call, 6L),
  identical(
    vapply(progress_events, `[[`, character(1L), "event"),
    c(
      "cold_started", "cold_complete",
      rep(c("warmup_started", "warmup_complete"), 2L),
      rep(c("timed_started", "timed_complete"), 3L)
    )
  ),
  identical(
    vapply(progress_events, `[[`, integer(1L), "index"),
    c(1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 2L, 2L, 3L, 3L)
  ),
  all(vapply(
    progress_events[c(2L, 4L, 6L, 8L, 10L, 12L)],
    function(event) identical(event$seconds, 0.25),
    logical(1L)
  ))
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

# Inventory checks must surround each sample but never enter its duration.
guard_clock <- 0
guard_events <- character()
guarded <- benchmark_time_runs(
  cold_run = function() { guard_clock <<- guard_clock + 2; TRUE },
  timed_run = function() { guard_clock <<- guard_clock + 2; TRUE },
  warmups = 1L, timed_runs = 2L, summarize = summarize,
  clock = function() guard_clock,
  guard = function(context) {
    guard_events <<- c(guard_events, context)
    guard_clock <<- guard_clock + 100
  }
)
stopifnot(identical(guarded$cold_seconds, 2),
          identical(guarded$warm$runs_seconds, c(2, 2)),
          length(guard_events) == 8L)
error <- tryCatch(benchmark_time_runs(
  cold_run = function() TRUE, timed_run = function() TRUE,
  warmups = 0L, timed_runs = 1L, summarize = summarize,
  guard = function(context) {
    if (identical(context, "after retained sample 1")) stop("contended")
  }
), error = identity)
stopifnot(inherits(error, "error"),
          identical(conditionMessage(error), "contended"))

message("Benchmark timing/collection self-tests passed.")
