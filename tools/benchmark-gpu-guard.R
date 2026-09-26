benchmark_validate_idle_gpu_requirement <- function(profile, backends, required) {
  if (identical(profile, "full") && any(backends != "base") &&
      !isTRUE(required)) {
    stop("Full retained CUDA benchmarks require idle-GPU inspection; use ",
         "a smoke profile for non-retained experiments.", call. = FALSE)
  }
  invisible(TRUE)
}

benchmark_parse_compute_pids <- function(lines) {
  lines <- trimws(as.character(lines))
  if (anyNA(lines)) {
    stop("Could not parse GPU compute process inventory; refusing to assume ",
         "the GPU is idle.", call. = FALSE)
  }
  lines <- lines[nzchar(lines)]
  if (!length(lines) || identical(lines, "No running processes found")) {
    return(integer())
  }
  valid <- grepl("^[1-9][0-9]*$", lines)
  values <- suppressWarnings(as.numeric(lines))
  if (!all(valid) || any(!is.finite(values)) ||
      any(values > .Machine$integer.max)) {
    stop("Could not parse GPU compute process inventory; refusing to assume ",
         "the GPU is idle.", call. = FALSE)
  }
  sort(unique(as.integer(values)))
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
  valid_pid <- function(value) {
    is.numeric(value) && all(is.finite(value)) &&
      all(value > 0 & value <= .Machine$integer.max & value == floor(value))
  }
  if (!valid_pid(pids) || length(current_pid) != 1L || !valid_pid(current_pid)) {
    stop("Invalid GPU compute process inventory; refusing to assume the GPU ",
         "is idle.", call. = FALSE)
  }
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
