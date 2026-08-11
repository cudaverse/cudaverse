benchmark_checkpoint_previous <- function(path) paste0(path, ".previous")

benchmark_checkpoint_scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value) || is.na(value[[1L]])) default else value[[1L]]
}

benchmark_checkpoint_case_ids <- function(cases) {
  if (is.data.frame(cases)) return(as.character(cases$case_id))
  vapply(
    cases,
    function(value) as.character(benchmark_checkpoint_scalar(value$case_id, "")),
    character(1L)
  )
}

benchmark_checkpoint_case_rows <- function(cases) {
  if (is.data.frame(cases)) {
    return(lapply(seq_len(nrow(cases)), function(index) {
      as.list(cases[index, , drop = FALSE])
    }))
  }
  cases
}

benchmark_checkpoint_case_contract_same <- function(existing, expected) {
  left <- benchmark_checkpoint_case_rows(existing)
  right <- benchmark_checkpoint_case_rows(expected)
  if (length(left) != length(right)) return(FALSE)
  fields <- sort(unique(c(
    unlist(lapply(left, names), use.names = FALSE),
    unlist(lapply(right, names), use.names = FALSE)
  )))
  normalize <- function(rows) {
    lapply(rows, function(row) {
      setNames(vapply(fields, function(field) {
        as.character(benchmark_checkpoint_scalar(row[[field]], "<absent>"))
      }, character(1L)), fields)
    })
  }
  identical(normalize(left), normalize(right))
}

benchmark_checkpoint_case_complete <- function(value, backends) {
  if (!is.list(value) || !is.list(value$backends) ||
      !setequal(names(value$backends), backends)) {
    return(FALSE)
  }
  all(vapply(backends, function(backend) {
    result <- value$backends[[backend]]
    identical(benchmark_checkpoint_scalar(result$status, ""), "complete") &&
      isTRUE(as.logical(benchmark_checkpoint_scalar(
        result$validation$passed, FALSE
      )))
  }, logical(1L)))
}

validate_benchmark_resume <- function(existing, expected) {
  failures <- character()
  require_same <- function(left, right, message) {
    if (!identical(left, right)) failures <<- c(failures, message)
  }
  require_same(
    benchmark_checkpoint_scalar(existing$schema, ""),
    "cudaverse-benchmark/1",
    "existing report schema is not cudaverse-benchmark/1"
  )
  require_same(
    benchmark_checkpoint_scalar(existing$profile, ""),
    benchmark_checkpoint_scalar(expected$profile, ""),
    "benchmark profile changed"
  )
  require_same(
    benchmark_checkpoint_scalar(existing$source$commit, ""),
    benchmark_checkpoint_scalar(expected$source$commit, ""),
    "source commit changed"
  )
  if (isTRUE(as.logical(benchmark_checkpoint_scalar(
    existing$source$tracked_dirty, TRUE
  )))) {
    failures <- c(failures, "existing report source was dirty")
  }
  if (isTRUE(as.logical(benchmark_checkpoint_scalar(
    expected$source$tracked_dirty, TRUE
  )))) {
    failures <- c(failures, "current benchmark source is dirty")
  }
  for (field in c("R", "cudaverse", "torch")) {
    require_same(
      benchmark_checkpoint_scalar(existing$software[[field]], "<absent>"),
      benchmark_checkpoint_scalar(expected$software[[field]], "<absent>"),
      paste(field, "software identity changed")
    )
  }
  require_same(
    unlist(existing$hardware$nvidia_smi, recursive = TRUE, use.names = FALSE),
    unlist(expected$hardware$nvidia_smi, recursive = TRUE, use.names = FALSE),
    "GPU identity changed"
  )
  require_same(
    unlist(existing$contract$backends, recursive = TRUE, use.names = FALSE),
    unlist(expected$contract$backends, recursive = TRUE, use.names = FALSE),
    "benchmark backend order changed"
  )
  if (!benchmark_checkpoint_case_contract_same(
    existing$contract$cases, expected$contract$cases
  )) failures <- c(failures, "benchmark case contract changed")
  if (length(failures)) {
    stop(
      "Benchmark resume refused:\n- ",
      paste(unique(failures), collapse = "\n- "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

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
