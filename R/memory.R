.cuda_memory_fields <- c(
  "available", "total_bytes", "free_bytes", "used_bytes",
  "allocated_bytes", "allocated_peak_bytes", "reserved_bytes",
  "reserved_peak_bytes", "reason"
)

.cuda_memory_result <- function(snapshot, selection, error = NULL) {
  missing <- setdiff(.cuda_memory_fields, names(snapshot))
  if (length(missing)) {
    snapshot <- .backend_empty_memory_info("invalid_backend_report")
    error <- sprintf(
      "Backend `%s` omitted memory field(s): %s.",
      selection$backend,
      paste(missing, collapse = ", ")
    )
  }
  valid_header <- is.logical(snapshot$available) &&
    length(snapshot$available) == 1L && !is.na(snapshot$available) &&
    is.character(snapshot$reason) && length(snapshot$reason) == 1L &&
    !is.na(snapshot$reason) && nzchar(snapshot$reason)
  if (!valid_header) {
    snapshot <- .backend_empty_memory_info("invalid_backend_report")
    error <- sprintf(
      "Backend `%s` returned invalid memory availability metadata.",
      selection$backend
    )
  }
  numeric_fields <- setdiff(
    .cuda_memory_fields,
    c("available", "reason")
  )
  for (field in numeric_fields) {
    value <- snapshot[[field]]
    valid <- is.numeric(value) && length(value) == 1L &&
      (is.na(value) || (is.finite(value) && value >= 0))
    if (!valid) {
      snapshot <- .backend_empty_memory_info("invalid_backend_report")
      error <- sprintf(
        "Backend `%s` returned an invalid `%s` value.",
        selection$backend,
        field
      )
      break
    }
  }
  structure(
    c(
      list(
        requested_device = selection$requested_device,
        device = selection$device,
        backend = selection$backend,
        selection_reason = selection$selection_reason,
        fallback = selection$fallback
      ),
      snapshot[.cuda_memory_fields],
      list(error = error)
    ),
    class = "cuda_memory_info"
  )
}

#' Inspect CUDA memory
#'
#' Reports physical device memory when the selected backend exposes it and
#' allocator-owned current and peak bytes when those counters are available.
#' The native backend reports physical CUDA-driver memory plus allocations
#' owned by cudaverse. The optional torch backend reports its allocator's
#' allocated and reserved bytes. CPU selection returns an unavailable report
#' rather than pretending host RAM is CUDA memory.
#'
#' This function does not reset allocator peaks or retain a user tensor. The
#' first CUDA selection in an R session can run the small runtime self-test, so
#' native peak bytes can include its released temporary allocations. An
#' automatic request is safe on a machine without CUDA and records the CPU
#' fallback. An explicit `device = "cuda"` request remains strict.
#'
#' @param device Requested device: `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A `cuda_memory_info` list with selection metadata, physical
#'   `total_bytes`, `free_bytes`, and `used_bytes`, allocator
#'   `allocated_bytes`, `allocated_peak_bytes`, `reserved_bytes`, and
#'   `reserved_peak_bytes`, plus `reason` and any captured `error`. Unsupported
#'   counters are `NA_real_` rather than estimated.
#' @export
#' @examples
#' cuda_memory_info("cpu")
#' cuda_memory_info("auto")
cuda_memory_info <- function(device = c("auto", "cuda", "cpu")) {
  selection <- cuda_select_device(match.arg(device))
  backend <- selection$backend
  if (!.backend_has_operation(backend, "memory_info")) {
    return(.cuda_memory_result(
      .backend_empty_memory_info("backend_memory_unsupported"),
      selection
    ))
  }
  error <- NULL
  snapshot <- tryCatch(
    .backend_call(backend, "memory_info"),
    error = function(condition) {
      error <<- conditionMessage(condition)
      .backend_empty_memory_info("memory_query_failed")
    }
  )
  .cuda_memory_result(snapshot, selection, error)
}

.format_memory_bytes <- function(bytes) {
  if (!is.numeric(bytes) || length(bytes) != 1L || is.na(bytes)) {
    return("unknown")
  }
  units <- c("B", "KiB", "MiB", "GiB", "TiB")
  unit <- min(length(units), floor(log(max(bytes, 1), 1024)) + 1L)
  sprintf("%.2f %s", bytes / 1024^(unit - 1L), units[[unit]])
}

#' @export
print.cuda_memory_info <- function(x, ...) {
  cat(sprintf(
    "<cuda_memory_info available=%s device=%s backend=%s reason=%s>\n",
    x$available,
    x$device,
    x$backend,
    x$reason
  ))
  if (isTRUE(x$available)) {
    if (!is.na(x$total_bytes)) {
      cat(sprintf(
        "Physical: %s used / %s total (%s free)\n",
        .format_memory_bytes(x$used_bytes),
        .format_memory_bytes(x$total_bytes),
        .format_memory_bytes(x$free_bytes)
      ))
    }
    if (!is.na(x$allocated_bytes)) {
      cat(sprintf(
        "Allocator: %s current / %s peak",
        .format_memory_bytes(x$allocated_bytes),
        .format_memory_bytes(x$allocated_peak_bytes)
      ))
      if (!is.na(x$reserved_bytes)) {
        cat(sprintf(
          "; %s reserved / %s peak",
          .format_memory_bytes(x$reserved_bytes),
          .format_memory_bytes(x$reserved_peak_bytes)
        ))
      }
      cat("\n")
    }
  }
  if (is.character(x$error) && length(x$error) == 1L && nzchar(x$error)) {
    cat("Error: ", x$error, "\n", sep = "")
  }
  invisible(x)
}
