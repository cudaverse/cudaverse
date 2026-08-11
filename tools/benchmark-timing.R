benchmark_time_runs <- function(cold_run, timed_run, warmups, timed_runs,
                                summarize, collect = NULL,
                                clock = function() as.numeric(Sys.time())) {
  cold_start <- clock()
  cold_value <- cold_run()
  cold_seconds <- clock() - cold_start
  cold_value <- NULL
  invisible(gc(FALSE))

  for (index in seq_len(warmups)) {
    warm_value <- timed_run()
    warm_value <- NULL
    invisible(gc(FALSE))
  }

  values <- numeric(timed_runs)
  observations <- if (is.null(collect)) NULL else vector("list", timed_runs)
  last <- NULL
  for (index in seq_len(timed_runs)) {
    start <- clock()
    value <- timed_run()
    values[[index]] <- clock() - start
    if (!is.null(collect)) observations[[index]] <- collect(value)
    if (index == timed_runs) last <- value
    value <- NULL
    if (index != timed_runs) invisible(gc(FALSE))
  }

  list(
    cold_seconds = cold_seconds,
    warm = summarize(values),
    observations = observations,
    last = last
  )
}
