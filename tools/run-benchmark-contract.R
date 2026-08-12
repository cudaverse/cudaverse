required_packages <- c("cudaverse", "jsonlite", "Matrix")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1), quietly = TRUE
)]
if (length(missing_packages)) {
  stop(
    "Install required package(s): ", paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}
sys.source(
  file.path("tools", "benchmark-checkpoint-io.R"),
  envir = environment()
)
sys.source(
  file.path("tools", "benchmark-timing.R"),
  envir = environment()
)
sys.source(
  file.path("tools", "benchmark-memory.R"),
  envir = environment()
)

truthy <- function(name, default = "false") {
  tolower(Sys.getenv(name, unset = default)) %in% c("1", "true", "yes")
}

profile <- tolower(Sys.getenv("CUDAVERSE_BENCHMARK_PROFILE", unset = "smoke"))
if (!profile %in% c("smoke", "full")) {
  stop("CUDAVERSE_BENCHMARK_PROFILE must be smoke or full.", call. = FALSE)
}
backends <- trimws(strsplit(
  Sys.getenv("CUDAVERSE_BENCHMARK_BACKENDS", unset = "base,native,torch"),
  ",", fixed = TRUE
)[[1L]])
if (!length(backends) || any(!backends %in% c("base", "native", "torch")) ||
    anyDuplicated(backends)) {
  stop("Benchmark backends must be unique base/native/torch values.",
       call. = FALSE)
}
if (!identical(backends[[1L]], "base")) {
  stop("The base backend must run first to establish parity references.",
       call. = FALSE)
}

contract_path <- Sys.getenv(
  "CUDAVERSE_BENCHMARK_CONTRACT",
  unset = file.path("inst", "benchmarks", "contract.csv")
)
contract <- utils::read.csv(
  contract_path, stringsAsFactors = FALSE, check.names = FALSE,
  na.strings = c("", "NA")
)
cases <- contract[contract$profile == profile, , drop = FALSE]
if (!nrow(cases)) stop("The selected benchmark profile has no cases.")

output <- Sys.getenv(
  "CUDAVERSE_BENCHMARK_OUTPUT",
  unset = file.path(tempdir(), paste0("cudaverse-benchmark-", profile, ".json"))
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
resume <- truthy("CUDAVERSE_BENCHMARK_RESUME")
overwrite <- truthy("CUDAVERSE_BENCHMARK_OVERWRITE")
if (resume && overwrite) {
  stop(
    "CUDAVERSE_BENCHMARK_RESUME and CUDAVERSE_BENCHMARK_OVERWRITE cannot ",
    "both be true.", call. = FALSE
  )
}

elapsed <- function() as.numeric(Sys.time())
summary_times <- function(values) {
  list(
    median_seconds = unname(stats::median(values)),
    p95_seconds = unname(stats::quantile(values, 0.95, type = 8)),
    runs_seconds = unname(as.numeric(values))
  )
}

source_state <- function(path = ".") {
  commit <- system2("git", c("-C", path, "rev-parse", "HEAD"), stdout = TRUE)
  tracked <- system2(
    "git", c("-C", path, "status", "--porcelain", "--untracked-files=all"),
    stdout = TRUE
  )
  list(
    commit = unname(commit[[1L]]),
    tracked_dirty = length(tracked) > 0L
  )
}

source <- source_state(".")
if (isTRUE(source$tracked_dirty) &&
    !truthy("CUDAVERSE_BENCHMARK_ALLOW_DIRTY")) {
  stop(
    "Benchmark source has tracked changes. Commit them or explicitly set ",
    "CUDAVERSE_BENCHMARK_ALLOW_DIRTY=true for a non-candidate smoke run.",
    call. = FALSE
  )
}

installed_size <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) return(NA_real_)
  root <- find.package(package)
  files <- list.files(
    root, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE
  )
  sum(file.info(files)$size, na.rm = TRUE)
}

gpu_identity <- function() {
  tryCatch(
    system2(
      "nvidia-smi",
      c(
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE, stderr = TRUE
    ),
    error = function(error) conditionMessage(error)
  )
}

diagnostics_start <- elapsed()
diagnostics <- cudaverse::cuda_diagnostics()
diagnostics_cold_seconds <- elapsed() - diagnostics_start
backend_details <- function(backend) {
  if (identical(backend, "base")) return(list(available = TRUE))
  diagnostics$backend_diagnostics[[backend]]
}
for (backend in backends) {
  details <- backend_details(backend)
  if (!is.list(details) || !isTRUE(details$available)) {
    stop(
      "Requested benchmark backend is unavailable: ", backend, " (",
      if (is.list(details)) details$reason else "not registered", ")",
      call. = FALSE
    )
  }
  if (identical(backend, "native") && !isTRUE(details$auto_eligible)) {
    stop("Native benchmark backend is not auto-eligible.", call. = FALSE)
  }
}

native_factory <- if ("native" %in% backends) {
  cudaverse:::.native_backend_factory()
} else {
  NULL
}

with_backend <- function(backend, code) {
  old <- options(cudaverse.cuda_backends = backend)
  on.exit(options(old), add = TRUE)
  force(code)
}

backend_device <- function(backend) {
  if (identical(backend, "base")) "cpu" else "cuda"
}

synchronize <- function(backend) {
  if (identical(backend, "native")) {
    native_factory$synchronize()
  } else if (identical(backend, "torch")) {
    torch::cuda_synchronize()
  }
  invisible(TRUE)
}

measure_memory <- function(backend, run, progress = NULL) {
  if (!is.null(progress)) progress("memory_started", 1L, 1L, NA_real_)
  started <- elapsed()
  result <- benchmark_measure_memory(
    backend,
    run,
    synchronize = synchronize,
    memory_query = function(selected_backend) {
      with_backend(
        selected_backend,
        cudaverse::cuda_memory_info(backend_device(selected_backend))
      )
    },
    reset_native_peak = function() {
      cudaverse:::.native_memory_tracker(reset = TRUE)
      invisible(TRUE)
    }
  )
  if (!is.null(progress)) {
    progress("memory_complete", 1L, 1L, elapsed() - started)
  }
  result
}

provenance_payload <- function(x) {
  value <- cudaverse::cuda_provenance(x)
  list(
    schema = attr(value, "schema", exact = TRUE),
    compute_device = attr(value, "compute_device", exact = TRUE),
    stages = lapply(seq_len(nrow(value)), function(index) {
      as.list(value[index, , drop = FALSE])
    })
  )
}

seed_for <- function(case_id) {
  20260810L + sum(utf8ToInt(case_id))
}

make_dense <- function(case) {
  set.seed(seed_for(case$case_id))
  matrix(stats::rnorm(case$rows * case$columns), case$rows, case$columns)
}

make_matmul <- function(case) {
  set.seed(seed_for(case$case_id))
  left <- matrix(
    stats::rnorm(case$rows * case$columns), case$rows, case$columns
  )
  right <- matrix(
    stats::rnorm(case$columns * case$columns),
    case$columns, case$columns
  )
  list(left = left, right = right, reference = left %*% right)
}

make_sparse <- function(case) {
  set.seed(seed_for(case$case_id))
  rows <- case$rows
  columns <- case$columns
  target <- max(2L * rows + columns, ceiling(rows * columns * case$density))
  remaining <- max(0L, target - 2L * rows - columns)
  first <- sample.int(columns, rows, replace = TRUE)
  offset <- sample.int(columns - 1L, rows, replace = TRUE)
  second <- ((first + offset - 1L) %% columns) + 1L
  i <- c(
    rep(seq_len(rows), each = 2L),
    sample.int(rows, columns, replace = TRUE),
    sample.int(rows, remaining, replace = TRUE)
  )
  j <- c(
    as.vector(rbind(first, second)),
    seq_len(columns),
    sample.int(columns, remaining, replace = TRUE)
  )
  values <- stats::rexp(length(i), rate = 1) + 0.05
  methods::as(Matrix::drop0(Matrix::sparseMatrix(
    i = i, j = j, x = values, dims = c(rows, columns), giveCsparse = TRUE
  )), "dgCMatrix")
}

matmul_case <- function(case, backend, source) {
  left <- source$left
  right <- source$right
  reference <- source$reference
  device <- backend_device(backend)

  with_backend(backend, {
    host_run <- function() {
      x <- cudaverse::cuda_tensor(left, device = device, dtype = case$dtype)
      y <- cudaverse::cuda_tensor(right, device = device, dtype = case$dtype)
      result <- cudaverse::tensor_matmul(x, y)
      synchronize(backend)
      host <- cudaverse::to_cpu(result)
      synchronize(backend)
      list(host = host, result = result)
    }
    resident_x <- cudaverse::cuda_tensor(left, device = device, dtype = case$dtype)
    resident_y <- cudaverse::cuda_tensor(right, device = device, dtype = case$dtype)
    synchronize(backend)
    resident_run <- function() {
      result <- cudaverse::tensor_matmul(resident_x, resident_y)
      synchronize(backend)
      result
    }

    host_timing <- benchmark_time_runs(
      host_run, host_run, case$warmups, case$timed_runs,
      summarize = summary_times,
      progress = benchmark_progress_logger(
        case$case_id, backend, "host_boundary"
      )
    )
    resident_timing <- benchmark_time_runs(
      resident_run, resident_run, case$warmups, case$timed_runs,
      summarize = summary_times,
      progress = benchmark_progress_logger(
        case$case_id, backend, "resident_compute"
      )
    )
    actual <- host_timing$last$host
    scale <- max(1, max(abs(reference)))
    absolute <- max(abs(actual - reference))
    relative <- absolute / scale
    tolerance <- if (identical(case$dtype, "float32")) {
      list(rtol = 1e-5, atol = 1e-6)
    } else {
      list(rtol = 1e-8, atol = 1e-10)
    }
    validation <- list(
      max_absolute_error = absolute,
      max_relative_error = relative,
      rtol = tolerance$rtol,
      atol = tolerance$atol,
      passed = absolute <= tolerance$atol + tolerance$rtol * scale
    )
    provenance <- provenance_payload(resident_timing$last)
    memory <- measure_memory(
      backend, host_run,
      progress = benchmark_progress_logger(case$case_id, backend, "memory")
    )
    host_timing$last <- NULL
    resident_timing$last <- NULL
    rm(resident_x, resident_y)
    invisible(gc(FALSE))

    list(
      status = "complete",
      cold_seconds = list(
        host_boundary = host_timing$cold_seconds,
        resident_compute = resident_timing$cold_seconds
      ),
      warm = list(
        host_boundary = host_timing$warm,
        resident_compute = resident_timing$warm
      ),
      transfer = list(
        included_boundary = "host matrices -> tensors -> matmul -> host matrix",
        excluded_boundary = "resident tensors -> resident tensor",
        separately_measured = TRUE
      ),
      validation = validation,
      provenance = provenance,
      memory = memory
    )
  })
}

pipeline_once <- function(case, backend, source, preloaded = NULL) {
  device <- backend_device(backend)
  start <- elapsed()
  if (identical(case$family, "sparse_pca_knn")) {
    sparse <- if (is.null(preloaded)) {
      cudaverse::cuda_sparse(source, device = device, format = "csr")
    } else {
      preloaded
    }
    synchronize(backend)
    transfer_end <- elapsed()
    normalized <- cudaverse::sparse_normalize(
      sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
    )
    synchronize(backend)
    normalization_end <- elapsed()
    pca_input <- normalized
  } else {
    sparse <- NULL
    normalized <- NULL
    transfer_end <- start
    normalization_end <- start
    pca_input <- source
  }
  fit <- cudaverse::cuda_pca(
    pca_input, n_components = case$components, device = device
  )
  synchronize(backend)
  pca_end <- elapsed()
  neighbors <- cudaverse::cuda_knn(
    fit$x, k = case$k, batch_size = min(256L, case$rows), device = device
  )
  synchronize(backend)
  end <- elapsed()
  list(
    value = list(sparse = sparse, normalized = normalized,
                 pca = fit, knn = neighbors),
    seconds = c(
      explicit_transfer = transfer_end - start,
      normalization = normalization_end - transfer_end,
      pca = pca_end - normalization_end,
      knn = end - pca_end,
      full_pipeline = end - start
    )
  )
}

pipeline_reference <- function(value) {
  list(
    normalized = if (is.null(value$normalized)) NULL else
      as.matrix(cudaverse::to_dgCMatrix(value$normalized)),
    pca = list(
      sdev = unname(value$pca$sdev),
      rotation = unname(value$pca$rotation),
      x = unname(value$pca$x)
    ),
    knn = list(
      index = unname(value$knn$index),
      distance = unname(value$knn$distance)
    )
  )
}

pipeline_validation <- function(value, reference) {
  rotation <- unname(value$pca$rotation)
  scores <- unname(value$pca$x)
  rank_threshold <- max(reference$pca$sdev) *
    max(nrow(reference$pca$x), nrow(reference$pca$rotation)) *
    .Machine$double.eps
  effective_rank <- max(1L, sum(reference$pca$sdev > rank_threshold))
  components <- seq_len(effective_rank)
  projector_error <- max(abs(
    tcrossprod(rotation[, components, drop = FALSE]) -
      tcrossprod(reference$pca$rotation[, components, drop = FALSE])
  ))
  reconstruction <- scores[, components, drop = FALSE] %*%
    t(rotation[, components, drop = FALSE])
  reference_reconstruction <-
    reference$pca$x[, components, drop = FALSE] %*%
    t(reference$pca$rotation[, components, drop = FALSE])
  reconstruction_scale <- max(1, max(abs(reference_reconstruction)))
  reconstruction_error <- max(abs(reconstruction - reference_reconstruction))
  indices_identical <- identical(unname(value$knn$index), reference$knn$index)
  distance_scale <- max(1, max(abs(reference$knn$distance)))
  distance_error <- max(abs(
    unname(value$knn$distance) - reference$knn$distance
  ))
  normalized_error <- if (is.null(reference$normalized)) 0 else max(abs(
    as.matrix(cudaverse::to_dgCMatrix(value$normalized)) -
      reference$normalized
  ))
  normalized_scale <- if (is.null(reference$normalized)) 1 else
    max(1, max(abs(reference$normalized)))
  list(
    normalized_max_relative_error = normalized_error / normalized_scale,
    pca_effective_rank = effective_rank,
    pca_projector_max_absolute_error = projector_error,
    pca_reconstruction_max_relative_error =
      reconstruction_error / reconstruction_scale,
    knn_indices_identical = indices_identical,
    knn_distance_max_relative_error = distance_error / distance_scale,
    passed = normalized_error / normalized_scale <= 1e-10 &&
      projector_error <= 1e-8 &&
      reconstruction_error / reconstruction_scale <= 1e-8 &&
      indices_identical && distance_error / distance_scale <= 1e-8
  )
}

pipeline_case <- function(case, backend, source, reference) {
  with_backend(backend, {
    included_run <- function() pipeline_once(case, backend, source)
    preloaded <- if (identical(case$family, "sparse_pca_knn")) {
      cudaverse::cuda_sparse(
        source, device = backend_device(backend), format = "csr"
      )
    } else {
      NULL
    }
    excluded_run <- function() pipeline_once(
      case, backend, source, preloaded = preloaded
    )
    included <- benchmark_time_runs(
      included_run, included_run, case$warmups, case$timed_runs,
      summarize = summary_times,
      collect = function(result) result$seconds,
      progress = benchmark_progress_logger(
        case$case_id, backend, "host_boundary"
      )
    )
    excluded <- if (is.null(preloaded)) NULL else benchmark_time_runs(
      excluded_run, excluded_run, case$warmups, case$timed_runs,
      summarize = summary_times,
      progress = benchmark_progress_logger(
        case$case_id, backend, "resident_continuation"
      )
    )
    value <- included$last$value
    validation <- if (is.null(reference)) {
      list(passed = TRUE, reference_backend = "self")
    } else {
      pipeline_validation(value, reference)
    }
    provenance <- list(
      pca = provenance_payload(value$pca),
      knn = provenance_payload(value$knn),
      normalized = if (is.null(value$normalized)) NULL else
        provenance_payload(value$normalized)
    )
    # Stage distributions are collected from the same synchronized timed runs
    # as the host-boundary distribution. This keeps every stage and its full
    # pipeline in one observation without executing the workload a second time.
    measurements <- do.call(rbind, included$observations)
    stage_times <- lapply(seq_len(ncol(measurements)), function(index) {
      summary_times(measurements[, index])
    })
    names(stage_times) <- colnames(measurements)
    memory <- measure_memory(
      backend, included_run,
      progress = benchmark_progress_logger(case$case_id, backend, "memory")
    )
    included$observations <- NULL
    included$last <- NULL
    if (!is.null(excluded)) excluded$last <- NULL
    preloaded <- NULL
    invisible(gc(FALSE))

    list(
      status = "complete",
      cold_seconds = list(
        host_boundary = included$cold_seconds,
        resident_continuation = if (is.null(excluded)) NA_real_ else
          excluded$cold_seconds
      ),
      warm = list(
        host_boundary = included$warm,
        resident_continuation = if (is.null(excluded)) {
          list(
            status = "not_separable",
            reason = paste(
              "Dense PCA accepts host data at the public boundary; its backend",
              "upload is included in PCA stage timing."
            )
          )
        } else {
          excluded$warm
        },
        stages = stage_times
      ),
      transfer = list(
        contract = case$transfer_contract,
        explicit_transfer_separately_measured = !is.null(excluded),
        note = if (is.null(excluded)) {
          paste(
            "Dense input upload is internal to the selected PCA backend and",
            "is not presented as a separately measured public-API duration."
          )
        } else {
          paste(
            "Host-to-sparse upload is timed separately; resident continuation",
            "begins from the preloaded cudasparse object."
          )
        }
      ),
      validation = validation,
      provenance = provenance,
      memory = memory,
      reference = pipeline_reference(value)
    )
  })
}

expected_report <- list(
  schema = "cudaverse-benchmark/1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  profile = profile,
  source = source,
  hardware = list(nvidia_smi = gpu_identity()),
  software = list(
    R = R.version.string,
    cudaverse = as.character(utils::packageVersion("cudaverse")),
    torch = if (requireNamespace("torch", quietly = TRUE))
      as.character(utils::packageVersion("torch")) else NA_character_,
    diagnostics = unclass(diagnostics)
  ),
  contract = list(
    path = contract_path,
    backends = backends,
    cases = cases,
    timing_clock = "wall clock with backend synchronization",
    stage_sampling = paste(
      "pipeline stages are collected from the same synchronized timed",
      "host-boundary runs"
    ),
    memory_sampling = paste(
      "one separate instrumented execution after timing; allocator tracking",
      "is excluded from retained timing samples"
    ),
    cold_runtime_diagnostics_seconds = diagnostics_cold_seconds,
    cold_scope = paste(
      "Cold case time is the first workload execution after package loading",
      "and the required diagnostics/self-test. Runtime diagnostics are",
      "reported separately; R process startup and package installation are",
      "outside the benchmark boundary."
    ),
    execution_order = paste(
      "cold host boundary, warmups/timed host boundary, cold resident",
      "continuation when separable, warmups/timed resident continuation"
    ),
    tolerances = list(
      float32 = list(rtol = 1e-5, atol = 1e-6),
      float64 = list(rtol = 1e-10, heavy_rtol = 1e-8),
      pca = "projector and reconstruction, heavy rtol 1e-8",
      knn = "exact indices; distance rtol 1e-8; ties by original row"
    )
  ),
  installed_size_bytes = list(
    cudaverse = installed_size("cudaverse"),
    torch = installed_size("torch"),
    bundled_cuda_runtime = 0
  ),
  cases = list(),
  complete = FALSE
)

previous_output <- benchmark_checkpoint_previous(output)
if (resume) {
  if (!benchmark_checkpoint_valid(output)) {
    recover_benchmark_checkpoint(output)
  }
  report <- jsonlite::read_json(output, simplifyVector = FALSE)
  validate_benchmark_resume(report, expected_report)
  reusable_cases <- names(report$cases)[vapply(
    report$cases,
    benchmark_checkpoint_case_complete,
    logical(1L),
    backends = backends
  )]
  report$complete <- FALSE
  history <- report$contract$resume_history
  if (is.null(history)) history <- list()
  history[[length(history) + 1L]] <- list(
    resumed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    reused_complete_cases = unname(reusable_cases)
  )
  report$contract$resume_history <- history
  message(
    "Validated benchmark resume; reusing ", length(reusable_cases),
    " complete case(s)."
  )
} else {
  existing_paths <- c(output, previous_output)[file.exists(c(output, previous_output))]
  if (length(existing_paths) && !overwrite) {
    stop(
      "Benchmark output already exists. Set CUDAVERSE_BENCHMARK_RESUME=true ",
      "for a validated resume or CUDAVERSE_BENCHMARK_OVERWRITE=true to ",
      "replace it.", call. = FALSE
    )
  }
  if (overwrite && length(existing_paths)) {
    failed <- vapply(existing_paths, unlink, integer(1L), force = TRUE)
    if (any(failed != 0L)) {
      stop("Could not remove existing benchmark output.", call. = FALSE)
    }
  }
  report <- expected_report
}

write_report <- function() {
  report$generated_at_utc <<- format(Sys.time(), tz = "UTC", usetz = TRUE)
  write_benchmark_checkpoint(report, output)
}

for (row in seq_len(nrow(cases))) {
  case <- as.list(cases[row, , drop = FALSE])
  case$rows <- as.integer(case$rows)
  case$columns <- as.integer(case$columns)
  case$warmups <- as.integer(case$warmups)
  case$timed_runs <- as.integer(case$timed_runs)
  case$k <- if (is.na(case$k)) NA_integer_ else as.integer(case$k)
  case$components <- if (is.na(case$components)) NA_integer_ else
    as.integer(case$components)
  message("Running benchmark case ", case$case_id)
  if (resume && benchmark_checkpoint_case_complete(
    report$cases[[case$case_id]], backends
  )) {
    message("  reusing complete checkpoint")
    next
  }
  report$cases[[case$case_id]] <- list(
    definition = case,
    backends = list()
  )
  benchmark_source <- if (identical(case$family, "matmul")) {
    make_matmul(case)
  } else if (identical(case$family, "dense_pca_knn")) {
    make_dense(case)
  } else if (identical(case$family, "sparse_pca_knn")) {
    make_sparse(case)
  } else {
    NULL
  }
  reference <- NULL
  for (backend in backends) {
    message("  backend: ", backend)
    result <- if (identical(case$family, "matmul")) {
      matmul_case(case, backend, benchmark_source)
    } else {
      pipeline_case(case, backend, benchmark_source, reference)
    }
    if (!isTRUE(result$validation$passed)) {
      stop(case$case_id, " failed parity on backend ", backend, ".")
    }
    if (!identical(case$family, "matmul") && identical(backend, "base")) {
      reference <- result$reference
    }
    result$reference <- NULL
    report$cases[[case$case_id]]$backends[[backend]] <- result
    write_report()
  }
  benchmark_source <- NULL
  reference <- NULL
  invisible(gc(FALSE))
}

report$complete <- TRUE
write_report()
finalize_benchmark_checkpoint(output)
message("Benchmark report complete: ", normalizePath(output, winslash = "/"))
