benchmark_checkpoint_previous <- function(path) paste0(path, ".previous")

benchmark_checkpoint_valid <- function(path) {
  if (!file.exists(path)) return(FALSE)
  value <- tryCatch(
    jsonlite::read_json(path, simplifyVector = FALSE),
    error = function(...) NULL
  )
  if (is.null(value)) return(FALSE)
  schema <- unlist(value$schema, recursive = TRUE, use.names = FALSE)
  length(schema) == 1L && identical(schema[[1L]], "cudaverse-benchmark/1")
}

write_benchmark_checkpoint <- function(value, path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Benchmark checkpointing requires jsonlite.", call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  staging <- paste0(path, ".writing-", Sys.getpid())
  previous <- benchmark_checkpoint_previous(path)
  on.exit(unlink(staging, force = TRUE), add = TRUE)

  jsonlite::write_json(
    value, staging, auto_unbox = TRUE, pretty = TRUE, digits = 16,
    null = "null", na = "null"
  )
  if (!benchmark_checkpoint_valid(staging)) {
    stop("The staged benchmark checkpoint is not valid JSON.", call. = FALSE)
  }

  if (file.exists(previous) && unlink(previous, force = TRUE) != 0L) {
    stop("Could not remove the older benchmark checkpoint.", call. = FALSE)
  }
  if (file.exists(path) && !file.rename(path, previous)) {
    stop("Could not rotate the current benchmark checkpoint.", call. = FALSE)
  }
  if (!file.rename(staging, path)) {
    if (file.exists(previous)) file.rename(previous, path)
    stop("Could not install the staged benchmark checkpoint.", call. = FALSE)
  }
  if (!benchmark_checkpoint_valid(path)) {
    if (file.exists(previous)) {
      unlink(path, force = TRUE)
      file.rename(previous, path)
    }
    stop("The installed benchmark checkpoint is not valid JSON.",
         call. = FALSE)
  }
  invisible(path)
}

recover_benchmark_checkpoint <- function(path) {
  if (benchmark_checkpoint_valid(path)) return("current")
  previous <- benchmark_checkpoint_previous(path)
  if (!benchmark_checkpoint_valid(previous)) {
    stop("Neither the current nor previous benchmark checkpoint is valid.",
         call. = FALSE)
  }
  if (file.exists(path) && unlink(path, force = TRUE) != 0L) {
    stop("Could not remove the invalid benchmark checkpoint.", call. = FALSE)
  }
  if (!file.rename(previous, path) || !benchmark_checkpoint_valid(path)) {
    stop("Could not recover the previous benchmark checkpoint.",
         call. = FALSE)
  }
  "previous"
}

finalize_benchmark_checkpoint <- function(path) {
  if (!benchmark_checkpoint_valid(path)) {
    stop("Cannot finalize an invalid benchmark checkpoint.", call. = FALSE)
  }
  value <- jsonlite::read_json(path, simplifyVector = FALSE)
  complete <- isTRUE(as.logical(unlist(
    value$complete, recursive = TRUE, use.names = FALSE
  )[[1L]]))
  if (!complete) {
    stop("Cannot finalize an incomplete benchmark report.", call. = FALSE)
  }
  previous <- benchmark_checkpoint_previous(path)
  if (file.exists(previous) && unlink(previous, force = TRUE) != 0L) {
    stop("Could not remove the previous completed checkpoint.", call. = FALSE)
  }
  invisible(path)
}
