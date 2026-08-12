arguments <- commandArgs(trailingOnly = TRUE)
all_arguments <- commandArgs(trailingOnly = FALSE)
script_argument <- grep("^--file=", all_arguments, value = TRUE)
if (length(script_argument) != 1L) {
  stop("Run this contract with Rscript.", call. = FALSE)
}
script_path <- normalizePath(
  sub("^--file=", "", script_argument),
  winslash = "/",
  mustWork = TRUE
)

session_worker <- length(arguments) >= 1L &&
  identical(arguments[[1L]], "--worker")

if (session_worker) {
  if (length(arguments) != 4L) {
    stop("The isolated session worker arguments are incomplete.",
         call. = FALSE)
  }
  session_library <- arguments[[2L]]
  expected_version <- arguments[[3L]]
  session_commit <- arguments[[4L]]
  if (!nzchar(session_library) || !dir.exists(session_library)) {
    stop("The isolated session library is missing.", call. = FALSE)
  }
  .libPaths(c(session_library, .libPaths()))
  suppressPackageStartupMessages(library(cudaverse))

  actual_version <- as.character(utils::packageVersion("cudaverse"))
  if (!identical(actual_version, expected_version)) {
    stop(
      sprintf(
        "Installed version mismatch: expected %s, found %s.",
        expected_version,
        actual_version
      ),
      call. = FALSE
    )
  }

  options(cudaverse.cuda_backends = "native")
  diagnostics <- cudaverse::cuda_diagnostics()
  native <- diagnostics$backend_diagnostics$native
  if (!identical(diagnostics$selected_backend, "native") ||
      !isTRUE(native$auto_eligible) ||
      !isTRUE(native$self_test$passed)) {
    stop("Native CUDA did not pass fresh-session selection.", call. = FALSE)
  }

  factory <- getFromNamespace(".native_backend_factory", "cudaverse")()
  tracker <- getFromNamespace(".native_memory_tracker", "cudaverse")
  backend_call <- getFromNamespace(".backend_call", "cudaverse")
  factory$synchronize()
  gc()
  baseline <- tracker(reset = TRUE)$current

  dense_values <- matrix(sin(seq_len(192L) / 9), 24L, 8L)
  dense <- cudaverse::cuda_tensor(
    dense_values,
    device = "cuda",
    dtype = "float64"
  )
  view <- cudaverse::tensor_reshape(dense, c(12L, 16L))
  if (!isTRUE(all.equal(
    as.vector(cudaverse::to_cpu(view)),
    as.vector(dense_values),
    tolerance = 0
  ))) {
    stop("Fresh-session dense view parity failed.", call. = FALSE)
  }

  sparse_values <- matrix(0, 24L, 8L)
  sparse_values[cbind(seq_len(24L), rep(seq_len(8L), 3L))] <-
    seq_len(24L) / 7
  sparse <- cudaverse::cuda_sparse(
    sparse_values,
    format = "csr",
    device = "cuda"
  )
  normalized <- cudaverse::sparse_normalize(
    sparse,
    margin = "rows",
    scale_factor = 1000,
    log1p = TRUE
  )
  fit <- cudaverse::cuda_pca(
    normalized,
    n_components = 3L,
    center = TRUE,
    scale. = FALSE,
    device = "cuda"
  )
  neighbors <- cudaverse::cuda_knn(
    fit$x,
    k = 3L,
    batch_size = 7L,
    device = "cuda"
  )
  if (!identical(dim(neighbors$index), c(24L, 3L)) ||
      any(!is.finite(neighbors$distance))) {
    stop("Fresh-session sparse/PCA/kNN workflow failed.", call. = FALSE)
  }

  memory_before_error <- cudaverse::cuda_memory_info("cuda")
  if (!isTRUE(memory_before_error$available) ||
      !identical(memory_before_error$backend, "native") ||
      memory_before_error$allocated_bytes <= baseline) {
    stop("Fresh-session memory telemetry is invalid.", call. = FALSE)
  }

  condition <- tryCatch(
    backend_call("native", "test_inject_cuda_error", 4096L),
    error = identity
  )
  if (!inherits(condition, "cudaverse_native_error") ||
      !identical(condition$operation, "test_inject_cuda_error")) {
    stop("Injected CUDA error did not retain its condition contract.",
         call. = FALSE)
  }

  rm(
    dense, view, sparse, normalized, fit, neighbors,
    memory_before_error, condition
  )
  gc()
  factory$synchronize()
  after_cleanup <- tracker()$current
  if (!identical(after_cleanup, baseline)) {
    stop(
      sprintf(
        "Fresh-session cleanup retained %.0f tracked bytes.",
        after_cleanup - baseline
      ),
      call. = FALSE
    )
  }

  left <- cudaverse::cuda_tensor(
    matrix(1:6, 2L, 3L), device = "cuda", dtype = "float64"
  )
  right <- cudaverse::cuda_tensor(
    matrix(1:6, 3L, 2L), device = "cuda", dtype = "float64"
  )
  product <- cudaverse::tensor_matmul(left, right)
  reference <- matrix(1:6, 2L, 3L) %*% matrix(1:6, 3L, 2L)
  if (!isTRUE(all.equal(
    cudaverse::to_cpu(product), reference, tolerance = 1e-10
  ))) {
    stop("Native backend was not reusable after the injected error.",
         call. = FALSE)
  }
  rm(left, right, product)
  gc()
  factory$synchronize()
  final <- tracker()$current
  if (!identical(final, baseline)) {
    stop(
      sprintf("Fresh-session final allocation is %.0f bytes above baseline.",
              final - baseline),
      call. = FALSE
    )
  }

  cat(sprintf(
    paste0(
      "CUDAVERSE_SESSION_OK pid=%s version=%s commit=%s ",
      "baseline=%.0f final=%.0f\n"
    ),
    Sys.getpid(),
    actual_version,
    session_commit,
    baseline,
    final
  ))
  quit(save = "no", status = 0L, runLast = FALSE)
}

if (!identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true")) {
  stop(
    "Set CUDAVERSE_NATIVE_TESTS=true to run the native session contract.",
    call. = FALSE
  )
}

source_path <- if (length(arguments)) arguments[[1L]] else "."
source_path <- normalizePath(source_path, winslash = "/", mustWork = TRUE)
report_path <- if (length(arguments) >= 2L) arguments[[2L]] else NULL
description_path <- file.path(source_path, "DESCRIPTION")
if (!file.exists(description_path)) {
  stop("The source path is not an R package.", call. = FALSE)
}
description <- read.dcf(description_path)
version <- unname(description[1L, "Version"])
commit <- tryCatch(
  system2(
    "git",
    c("-C", shQuote(source_path), "rev-parse", "HEAD"),
    stdout = TRUE,
    stderr = TRUE
  ),
  error = function(error) NA_character_
)
if (length(commit) != 1L || !grepl("^[0-9a-f]{40}$", commit)) {
  commit <- "unknown"
}
tracked_dirty <- tryCatch(
  length(system2(
    "git",
    c(
      "-C", shQuote(source_path), "status", "--porcelain",
      "--untracked-files=no"
    ),
    stdout = TRUE,
    stderr = TRUE
  )) > 0L,
  error = function(error) TRUE
)

session_library <- tempfile("cudaverse-session-library-")
dir.create(session_library)
r_binary <- file.path(R.home("bin"), "R")
rscript_binary <- file.path(R.home("bin"), "Rscript")
install_output <- system2(
  r_binary,
  c(
    "CMD", "INSTALL", "--no-multiarch", "--with-keep.source",
    "-l", shQuote(session_library), shQuote(source_path)
  ),
  stdout = TRUE,
  stderr = TRUE
)
install_status <- attr(install_output, "status")
if (!is.null(install_status) && install_status != 0L) {
  cat(install_output, sep = "\n")
  stop("Installing the isolated session candidate failed.", call. = FALSE)
}

session_pids <- integer()
session_outputs <- vector("list", 2L)
for (iteration in seq_len(2L)) {
  output <- system2(
    rscript_binary,
    c(
      "--vanilla",
      shQuote(script_path),
      "--worker",
      shQuote(session_library),
      shQuote(version),
      shQuote(commit)
    ),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    cat(output, sep = "\n")
    stop(sprintf("Isolated session %s failed.", iteration), call. = FALSE)
  }
  marker <- grep("^CUDAVERSE_SESSION_OK ", output, value = TRUE)
  if (length(marker) != 1L) {
    cat(output, sep = "\n")
    stop(sprintf("Isolated session %s did not emit evidence.", iteration),
         call. = FALSE)
  }
  pid <- sub(".* pid=([0-9]+) .*", "\\1", marker)
  if (!grepl("^[0-9]+$", pid)) {
    stop("The isolated session emitted an invalid process id.", call. = FALSE)
  }
  session_pids <- c(session_pids, as.integer(pid))
  session_outputs[[iteration]] <- marker
}
if (anyDuplicated(session_pids)) {
  stop("The session contract did not create two distinct processes.",
       call. = FALSE)
}

nvidia_smi <- Sys.which("nvidia-smi")
hardware <- NA_character_
if (nzchar(nvidia_smi)) {
  active <- tryCatch(
    system2(
      nvidia_smi,
      c("--query-compute-apps=pid", "--format=csv,noheader,nounits"),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(error) character()
  )
  active <- suppressWarnings(as.integer(trimws(active)))
  active <- active[!is.na(active)]
  if (any(session_pids %in% active)) {
    stop("An exited session still appears in the NVIDIA process table.",
         call. = FALSE)
  }
  hardware <- tryCatch(
    paste(system2(
      nvidia_smi,
      c(
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = FALSE
    ), collapse = "; "),
    error = function(error) NA_character_
  )
}

cat(unlist(session_outputs), sep = "\n")
cat(sprintf(
  "NATIVE_SESSION_CONTRACT_OK sessions=2 pids=%s commit=%s\n",
  paste(session_pids, collapse = ","),
  commit
))

if (!is.null(report_path)) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Writing session evidence requires `jsonlite`.", call. = FALSE)
  }
  parse_number <- function(marker, field) {
    value <- sub(
      sprintf(".* %s=([0-9]+).*", field),
      "\\1",
      marker
    )
    if (!grepl("^[0-9]+$", value)) {
      stop(sprintf("Session marker omitted `%s`.", field), call. = FALSE)
    }
    as.numeric(value)
  }
  sessions <- lapply(seq_along(session_outputs), function(index) {
    marker <- session_outputs[[index]]
    list(
      session = as.integer(index),
      pid = as.integer(session_pids[[index]]),
      version = version,
      commit = commit,
      baseline_bytes = parse_number(marker, "baseline"),
      final_bytes = parse_number(marker, "final"),
      process_absent_after_exit = TRUE,
      workflow = c(
        "native-self-test", "dense-shared-view", "sparse-normalize",
        "resident-pca-knn", "memory-telemetry", "injected-oom",
        "post-error-matmul", "allocator-cleanup", "clean-exit"
      )
    )
  })
  report <- list(
    schema = "cudaverse-native-session/1",
    generated_at_utc = format(
      Sys.time(), "%Y-%m-%d %H:%M:%S UTC", tz = "UTC"
    ),
    source = list(
      commit = commit,
      tracked_dirty = tracked_dirty,
      version = version
    ),
    hardware = list(nvidia_smi = hardware),
    contract = list(
      isolated_install = TRUE,
      vanilla_sessions = 2L,
      distinct_processes = !anyDuplicated(session_pids),
      native_backend_required = TRUE,
      injected_error = "CUDA_ERROR_OUT_OF_MEMORY",
      exact_allocator_cleanup = TRUE,
      exited_process_check = nzchar(nvidia_smi)
    ),
    sessions = sessions,
    passed = TRUE
  )
  report_path <- normalizePath(
    report_path,
    winslash = "/",
    mustWork = FALSE
  )
  dir.create(dirname(report_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(
    report,
    report_path,
    pretty = TRUE,
    auto_unbox = TRUE,
    digits = NA
  )
  cat("Session report written: ", report_path, "\n", sep = "")
}
