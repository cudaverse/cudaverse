benchmark_parse_compute_pids <- function(lines) {
  values <- suppressWarnings(as.integer(trimws(as.character(lines))))
  sort(unique(values[is.finite(values) & values > 0L]))
}

benchmark_gpu_compute_pids <- function(command = "nvidia-smi") {
  output <- tryCatch(
    system2(
      command,
      c(
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = identity
  )
  if (inherits(output, "error")) {
    stop(
      "Could not inspect active GPU compute processes: ",
      conditionMessage(output),
      call. = FALSE
    )
  }
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(
      "Could not inspect active GPU compute processes: ",
      paste(output, collapse = " "),
      call. = FALSE
    )
  }
  benchmark_parse_compute_pids(output)
}

benchmark_assert_idle_gpu <- function(
  context,
  pids = benchmark_gpu_compute_pids(),
  current_pid = Sys.getpid()
) {
  competing <- setdiff(as.integer(pids), as.integer(current_pid))
  if (length(competing)) {
    stop(
      "Retained benchmark requires an idle GPU; competing compute process ",
      "detected during ", context, ": ", paste(competing, collapse = ", "),
      ". Preserve the checkpoint and resume when the GPU is idle.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
