check_capability_matrix <- function(root = ".") {
  path <- file.path(root, "inst", "reports", "backend-capability-matrix.csv")
  if (!file.exists(path)) {
    stop("Capability matrix is missing: ", path, call. = FALSE)
  }

  matrix <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required_columns <- c(
    "api", "category", "cpu_base", "torch_cuda", "native_cuda",
    "result_location", "notes"
  )
  if (!identical(names(matrix), required_columns)) {
    stop("Capability matrix columns do not match the contract.", call. = FALSE)
  }
  if (anyDuplicated(matrix$api)) {
    stop("Capability matrix contains duplicate API rows.", call. = FALSE)
  }
  if (anyNA(matrix) || any(!nzchar(as.matrix(matrix)))) {
    stop("Capability matrix contains missing or empty fields.", call. = FALSE)
  }

  namespace <- readLines(file.path(root, "NAMESPACE"), warn = FALSE)
  export_lines <- grep("^export\\([^)]*\\)$", namespace, value = TRUE)
  exports <- sub("^export\\(([^)]*)\\)$", "\\1", export_lines)
  if (!setequal(matrix$api, exports)) {
    missing <- setdiff(exports, matrix$api)
    extra <- setdiff(matrix$api, exports)
    stop(
      paste0(
        "Capability matrix and NAMESPACE exports differ. Missing: ",
        paste(missing, collapse = ", "), "; extra: ",
        paste(extra, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  allowed_status <- c("direct", "hybrid", "cpu_only", "metadata", "probe", "host")
  for (column in c("cpu_base", "torch_cuda", "native_cuda")) {
    invalid <- setdiff(unique(matrix[[column]]), allowed_status)
    if (length(invalid)) {
      stop(
        "Invalid ", column, " status: ", paste(invalid, collapse = ", "),
        call. = FALSE
      )
    }
  }
  allowed_locations <- c(
    "host", "same_device", "host_with_native_cache", "backend_dependent"
  )
  invalid_locations <- setdiff(unique(matrix$result_location), allowed_locations)
  if (length(invalid_locations)) {
    stop(
      "Invalid result location: ", paste(invalid_locations, collapse = ", "),
      call. = FALSE
    )
  }

  invisible(matrix[order(matrix$api), , drop = FALSE])
}

if (sys.nframe() == 0L) {
  arguments <- commandArgs(trailingOnly = TRUE)
  root <- if (length(arguments)) arguments[[1L]] else "."
  result <- check_capability_matrix(root)
  message(
    "Capability matrix covers all ", nrow(result),
    " exported functions with valid backend statuses."
  )
}
