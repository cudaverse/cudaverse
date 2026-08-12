benchmark_progress_logger <- function(case_id, backend, scope,
                                      reporter = message) {
  force(case_id)
  force(backend)
  force(scope)
  function(event, index, total, seconds) {
    duration <- if (is.finite(seconds)) {
      paste0(
        " in ",
        trimws(formatC(seconds, digits = 6L, format = "fg")),
        " s"
      )
    } else {
      ""
    }
    reporter(paste0(
      "    progress: ", case_id, "/", backend, "/", scope, " ",
      event, " ", index, "/", total, duration
    ))
    invisible(NULL)
  }
}

benchmark_time_runs <- function(cold_run, timed_run, warmups, timed_runs,
                                summarize, collect = NULL,
                                clock = function() as.numeric(Sys.time()),
                                progress = NULL) {
  notify <- function(event, index, total, seconds = NA_real_) {
    if (!is.null(progress)) {
      progress(event, as.integer(index), as.integer(total), seconds)
    }
    invisible(NULL)
  }

  notify("cold_started", 1L, 1L)
  cold_start <- clock()
  cold_value <- cold_run()
  cold_seconds <- clock() - cold_start
  cold_value <- NULL
  invisible(gc(FALSE))
  notify("cold_complete", 1L, 1L, cold_seconds)

  for (index in seq_len(warmups)) {
    notify("warmup_started", index, warmups)
    warm_start <- clock()
    warm_value <- timed_run()
    warm_seconds <- clock() - warm_start
    warm_value <- NULL
    invisible(gc(FALSE))
    notify("warmup_complete", index, warmups, warm_seconds)
  }

  values <- numeric(timed_runs)
  observations <- if (is.null(collect)) NULL else vector("list", timed_runs)
  last <- NULL
  for (index in seq_len(timed_runs)) {
    notify("timed_started", index, timed_runs)
    start <- clock()
    value <- timed_run()
    values[[index]] <- clock() - start
    if (!is.null(collect)) observations[[index]] <- collect(value)
    if (index == timed_runs) last <- value
    value <- NULL
    if (index != timed_runs) invisible(gc(FALSE))
    notify("timed_complete", index, timed_runs, values[[index]])
  }

  list(
    cold_seconds = cold_seconds,
    warm = summarize(values),
    observations = observations,
    last = last
  )
}
