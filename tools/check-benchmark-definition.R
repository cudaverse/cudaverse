check_benchmark_definition <- function(root = ".") {
  path <- file.path(root, "inst", "benchmarks", "contract.csv")
  if (!file.exists(path)) {
    stop("Benchmark contract is missing: ", path, call. = FALSE)
  }
  contract <- utils::read.csv(
    path, stringsAsFactors = FALSE, check.names = FALSE,
    na.strings = c("", "NA")
  )
  required_columns <- c(
    "profile", "family", "case_id", "rows", "columns", "density",
    "dtype", "k", "components", "warmups", "timed_runs",
    "transfer_contract"
  )
  if (!identical(names(contract), required_columns)) {
    stop("Benchmark contract columns do not match schema.", call. = FALSE)
  }
  if (anyDuplicated(contract$case_id)) {
    stop("Benchmark case identifiers must be unique.", call. = FALSE)
  }
  required_text <- c(
    "profile", "family", "case_id", "dtype", "transfer_contract"
  )
  if (anyNA(contract[required_text]) ||
      any(!nzchar(as.matrix(contract[required_text])))) {
    stop("Benchmark contract contains missing text fields.", call. = FALSE)
  }
  if (anyNA(contract[c("rows", "columns", "warmups", "timed_runs")]) ||
      any(contract$rows < 1L) || any(contract$columns < 1L) ||
      any(contract$warmups < 1L) || any(contract$timed_runs < 1L)) {
    stop("Benchmark dimensions and repetition counts must be positive.",
         call. = FALSE)
  }
  if (!all(contract$profile %in% c("smoke", "full")) ||
      !all(contract$family %in%
             c("matmul", "dense_pca_knn", "sparse_pca_knn")) ||
      !all(contract$dtype %in% c("float32", "float64"))) {
    stop("Benchmark profile, family, or dtype is invalid.", call. = FALSE)
  }

  matmul <- contract$family == "matmul"
  dense <- contract$family == "dense_pca_knn"
  sparse <- contract$family == "sparse_pca_knn"
  if (any(!is.na(contract$density[!sparse])) ||
      any(is.na(contract$density[sparse])) ||
      any(contract$density[sparse] <= 0 | contract$density[sparse] > 1)) {
    stop("Density is required only for sparse cases and must be in (0, 1].",
         call. = FALSE)
  }
  if (any(!is.na(contract$k[matmul])) ||
      any(!is.na(contract$components[matmul])) ||
      any(is.na(contract$k[!matmul])) ||
      any(is.na(contract$components[!matmul]))) {
    stop("PCA/kNN fields must be present only for pipeline cases.",
         call. = FALSE)
  }
  if (any(contract$k[!matmul] >= contract$rows[!matmul]) ||
      any(contract$components[!matmul] >
            pmin(contract$columns[!matmul], contract$rows[!matmul] - 1L))) {
    stop("Pipeline k/components exceed the input dimensions.", call. = FALSE)
  }

  full <- contract[contract$profile == "full", , drop = FALSE]
  expected_matmul <- as.vector(outer(
    c("float32", "float64"), c(256L, 1024L, 4096L),
    function(dtype, size) paste0("matmul-", dtype, "-", size)
  ))
  expected_dense <- paste0(
    "dense-", c("1000x50", "10000x100", "50000x128")
  )
  expected_sparse <- paste0(
    "sparse-", c(
      "1000x50@0.10", "10000x100@0.03", "50000x128@0.01"
    )
  )
  if (!setequal(
    full$case_id,
    c(expected_matmul, expected_dense, expected_sparse)
  )) {
    stop("The full benchmark profile does not contain every required case.",
         call. = FALSE)
  }
  if (any(full$warmups != 5L) || any(full$timed_runs != 10L)) {
    stop("The full profile requires five warmups and ten timed runs.",
         call. = FALSE)
  }
  if (any(full$k[!is.na(full$k)] != 15L)) {
    stop("The full PCA/kNN profile requires k = 15.", call. = FALSE)
  }
  if (!all(c("smoke", "full") %in% contract$profile)) {
    stop("Both smoke and full benchmark profiles are required.",
         call. = FALSE)
  }
  invisible(contract)
}

if (sys.nframe() == 0L) {
  arguments <- commandArgs(trailingOnly = TRUE)
  root <- if (length(arguments)) arguments[[1L]] else "."
  result <- check_benchmark_definition(root)
  message(
    "Benchmark contract is valid: ", nrow(result), " cases (",
    sum(result$profile == "smoke"), " smoke, ",
    sum(result$profile == "full"), " full)."
  )
}
