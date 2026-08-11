if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("Matrix", quietly = TRUE)) {
  stop("The `jsonlite`, `digest`, and `Matrix` packages are required for Phase 4.")
}
if (!requireNamespace("cudaverse", quietly = TRUE)) {
  stop("Install the development cudaverse package first.")
}

integer_setting <- function(name, default, minimum = 1L) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, unset = default)))
  if (length(value) != 1L || is.na(value) || value < minimum) {
    stop("Invalid `", name, "` setting.")
  }
  value
}

parse_cases <- function(value) {
  fields <- strsplit(value, ",", fixed = TRUE)[[1L]]
  cases <- lapply(fields, function(field) {
    parts <- strsplit(tolower(trimws(field)), "x", fixed = TRUE)[[1L]]
    if (length(parts) != 3L) {
      stop("Invalid case `", field, "`; use rowsxcolumnsxdensity.")
    }
    rows <- suppressWarnings(as.integer(parts[[1L]]))
    columns <- suppressWarnings(as.integer(parts[[2L]]))
    density <- suppressWarnings(as.numeric(parts[[3L]]))
    if (anyNA(c(rows, columns, density)) || rows < 2L || columns < 2L ||
        density <= 0 || density > 1) {
      stop("Invalid case `", field, "`; density must be in (0, 1].")
    }
    list(rows = rows, columns = columns, density = density)
  })
  names(cases) <- vapply(
    cases,
    function(x) sprintf("%sx%s@%s", x$rows, x$columns, x$density),
    character(1)
  )
  cases
}

summary_times <- function(values) {
  list(
    median_seconds = unname(stats::median(values)),
    p95_seconds = unname(stats::quantile(values, 0.95, type = 8)),
    runs_seconds = unname(values)
  )
}

installed_size <- function(package) {
  path <- find.package(package)
  files <- list.files(
    path, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE
  )
  sum(file.info(files)$size, na.rm = TRUE)
}

gpu_line <- function() {
  tryCatch(
    system2(
      "nvidia-smi",
      c(
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(error) conditionMessage(error)
  )
}

source_state <- function(path = ".") {
  commit <- tryCatch(
    system2("git", c("-C", path, "rev-parse", "HEAD"), stdout = TRUE),
    error = function(error) "unavailable"
  )
  tracked_changes <- tryCatch(
    system2(
      "git",
      c("-C", path, "status", "--porcelain", "--untracked-files=no"),
      stdout = TRUE
    ),
    error = function(error) "unavailable"
  )
  list(
    commit = unname(commit[[1L]]),
    tracked_dirty = length(tracked_changes) > 0L
  )
}

provenance_payload <- function(x) {
  value <- cudaverse::cuda_provenance(x)
  list(
    schema = attr(value, "schema", exact = TRUE),
    compute_device = attr(value, "compute_device", exact = TRUE),
    stages = lapply(value, unname)
  )
}

make_sparse_input <- function(rows, columns, density, seed) {
  set.seed(seed)
  target <- max(2L * rows + columns, ceiling(rows * columns * density))
  remaining <- max(0L, target - 2L * rows - columns)
  first_column <- sample.int(columns, rows, replace = TRUE)
  second_offset <- sample.int(columns - 1L, rows, replace = TRUE)
  second_column <- ((first_column + second_offset - 1L) %% columns) + 1L
  i <- c(
    rep(seq_len(rows), each = 2L),
    sample.int(rows, columns, replace = TRUE),
    sample.int(rows, remaining, replace = TRUE)
  )
  j <- c(
    as.vector(rbind(first_column, second_column)),
    seq_len(columns),
    sample.int(columns, remaining, replace = TRUE)
  )
  values <- stats::rexp(length(i), rate = 1) + 0.05
  result <- Matrix::sparseMatrix(
    i = i,
    j = j,
    x = values,
    dims = c(rows, columns),
    dimnames = list(
      paste0("sample_", seq_len(rows)),
      paste0("feature_", seq_len(columns))
    ),
    giveCsparse = TRUE
  )
  methods::as(Matrix::drop0(result), "dgCMatrix")
}

strip_matrix <- function(x) {
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}

reference_payload <- function(value) {
  list(
    normalized = strip_matrix(as.matrix(
      cudaverse::to_dgCMatrix(value$normalized)
    )),
    pca = list(
      sdev = unname(value$pca$sdev),
      rotation = strip_matrix(value$pca$rotation),
      x = strip_matrix(value$pca$x)
    ),
    knn = list(
      index = unname(value$knn$index),
      distance = unname(value$knn$distance)
    )
  )
}

validation_payload <- function(value, reference) {
  normalized <- strip_matrix(as.matrix(
    cudaverse::to_dgCMatrix(value$normalized)
  ))
  normalized_error <- max(abs(normalized - reference$normalized))
  normalized_scale <- max(1, max(abs(reference$normalized)))

  actual_rotation <- strip_matrix(value$pca$rotation)
  actual_scores <- strip_matrix(value$pca$x)
  rank_threshold <- max(reference$pca$sdev) *
    max(nrow(reference$pca$x), nrow(reference$pca$rotation)) *
    .Machine$double.eps
  effective_rank <- sum(reference$pca$sdev > rank_threshold)
  stable_components <- seq_len(effective_rank)
  actual_projector <- tcrossprod(
    actual_rotation[, stable_components, drop = FALSE]
  )
  reference_projector <- tcrossprod(
    reference$pca$rotation[, stable_components, drop = FALSE]
  )
  actual_reconstruction <- actual_scores %*% t(actual_rotation)
  reference_reconstruction <- reference$pca$x %*%
    t(reference$pca$rotation)
  reconstruction_error <- max(abs(
    actual_reconstruction - reference_reconstruction
  ))
  reconstruction_scale <- max(1, max(abs(reference_reconstruction)))
  projector_error <- max(abs(actual_projector - reference_projector))

  distance <- unname(value$knn$distance)
  distance_error <- max(abs(distance - reference$knn$distance))
  distance_scale <- max(1, max(abs(reference$knn$distance)))
  indices_identical <- identical(
    unname(value$knn$index), reference$knn$index
  )
  passed <- normalized_error / normalized_scale <= 1e-10 &&
    projector_error <= 1e-8 &&
    reconstruction_error / reconstruction_scale <= 1e-8 &&
    indices_identical &&
    distance_error / distance_scale <= 1e-8

  list(
    normalized_max_absolute_error = normalized_error,
    normalized_max_relative_error = normalized_error / normalized_scale,
    pca_sdev_max_absolute_error = max(abs(
      unname(value$pca$sdev) - reference$pca$sdev
    )),
    pca_effective_rank = effective_rank,
    pca_rank_threshold = rank_threshold,
    pca_subspace_projector_max_absolute_error = projector_error,
    pca_reconstruction_max_absolute_error = reconstruction_error,
    pca_reconstruction_max_relative_error =
      reconstruction_error / reconstruction_scale,
    knn_indices_identical = indices_identical,
    knn_distance_max_absolute_error = distance_error,
    knn_distance_max_relative_error = distance_error / distance_scale,
    passed = passed
  )
}

factory <- cudaverse:::.native_backend_factory()
native_diagnostics <- factory$diagnostics()
if (!isTRUE(native_diagnostics$available)) {
  stop(
    "The native CUDA backend is unavailable: ",
    native_diagnostics$detection_error
  )
}

original_backend_options <- options(cudaverse.cuda_backends = NULL)
global_diagnostics <- cudaverse::cuda_diagnostics()
native_selection_diagnostics <-
  global_diagnostics$backend_diagnostics$native
auto_selection <- list(
  requested_device = "auto",
  selected_backend = global_diagnostics$selected_backend,
  auto_eligible_backends = global_diagnostics$auto_eligible_backends,
  selection_reason = global_diagnostics$auto_selection_reason,
  contract_schema = native_selection_diagnostics$contract_schema,
  capability_compatible =
    native_selection_diagnostics$capability_compatible,
  missing_auto_capabilities =
    native_selection_diagnostics$missing_auto_capabilities,
  runtime_complete = native_selection_diagnostics$runtime_complete,
  self_test = native_selection_diagnostics$self_test
)
auto_selection$passed <-
  identical(auto_selection$selected_backend, "native") &&
  "native" %in% auto_selection$auto_eligible_backends &&
  identical(auto_selection$selection_reason, "native_auto_eligible") &&
  identical(auto_selection$contract_schema, "cudaverse-backend/1") &&
  isTRUE(auto_selection$capability_compatible) &&
  !length(auto_selection$missing_auto_capabilities) &&
  isTRUE(auto_selection$runtime_complete) &&
  isTRUE(auto_selection$self_test$passed)
if (!isTRUE(auto_selection$passed)) {
  stop("The native backend did not pass every automatic-selection gate.")
}

warmup_runs <- integer_setting("CUDAVERSE_PHASE4_WARMUPS", "2")
timed_runs <- integer_setting("CUDAVERSE_PHASE4_TIMED_RUNS", "5")
k <- integer_setting("CUDAVERSE_PHASE4_K", "15")
batch_size <- integer_setting("CUDAVERSE_PHASE4_BATCH_SIZE", "256")
components <- integer_setting("CUDAVERSE_PHASE4_COMPONENTS", "20")
cases <- parse_cases(Sys.getenv(
  "CUDAVERSE_PHASE4_CASES",
  unset = "1000x50x0.10,5000x100x0.03,10000x128x0.01"
))
backends <- trimws(strsplit(Sys.getenv(
  "CUDAVERSE_PHASE4_BACKENDS", unset = "base,native,torch"
), ",", fixed = TRUE)[[1L]])
if (length(backends) < 1L || any(!backends %in% c("base", "native", "torch"))) {
  stop("Backends must be selected from base, native, and torch.")
}

output <- Sys.getenv(
  "CUDAVERSE_PHASE4_REPORT",
  unset = file.path(
    "inst", "reports", "native", "consolidation-rtx2000.json"
  )
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
baseline_path <- Sys.getenv(
  "CUDAVERSE_PHASE4_BASELINE",
  unset = file.path("inst", "reports", "native", "phase3-rtx2000.json")
)
if (!file.exists(baseline_path)) {
  stop("The Phase 3 benchmark baseline does not exist: ", baseline_path)
}
baseline <- jsonlite::read_json(baseline_path, simplifyVector = FALSE)
if (!identical(baseline$schema, "cudaverse-native-phase3/1")) {
  stop("The benchmark baseline is not a Phase 3 report.")
}
baseline_bytes <- readChar(
  baseline_path,
  nchars = file.info(baseline_path)$size,
  useBytes = TRUE
)
baseline_bytes <- gsub("\r\n", "\n", baseline_bytes, fixed = TRUE)
baseline_sha256 <- digest::digest(
  charToRaw(baseline_bytes), algo = "sha256", serialize = FALSE
)

report <- list(
  schema = "cudaverse-native-consolidation/1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source = list(cudaverse = source_state(".")),
  hardware = list(nvidia_smi = gpu_line()),
  software = list(
    R = R.version.string,
    cudaverse = as.character(utils::packageVersion("cudaverse")),
    native_diagnostics = native_diagnostics
  ),
  contract = list(
    warmup_runs = warmup_runs,
    timed_runs = timed_runs,
    k = k,
    requested_components = components,
    batch_size = batch_size,
    tensor_dtypes = c("float32", "float64"),
    sparse_pipeline_dtype = "float64",
    metric = "euclidean",
    normalization = "row sum to 1000 followed by log1p",
    full_pipeline_boundary = paste(
      "host dgCMatrix -> backend COO/CSR transfer -> sparse normalization ->",
      "PCA -> device-resident scores -> exact distance -> stable top-k ->",
      "host n-by-k result"
    ),
    native_selection = paste(
      "automatic; contract, capabilities, runtime components, and cached",
      "self-test passed"
    ),
    tolerances = list(
      normalization_relative = 1e-10,
      pca_projector_absolute = 1e-8,
      pca_reconstruction_relative = 1e-8,
      knn_distance_relative = 1e-8,
      knn_indices = "identical with original-row stable tie ordering"
    ),
    cases = cases,
    backends = backends
  ),
  installed_size_bytes = list(cudaverse = installed_size("cudaverse")),
  benchmarks = list(),
  benchmark_regression = NULL,
  hardware_validation = NULL,
  overall_pass = FALSE
)

write_report <- function() {
  report$generated_at_utc <<- format(Sys.time(), tz = "UTC", usetz = TRUE)
  jsonlite::write_json(
    report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
    null = "null", na = "null"
  )
}

backend_diagnostics <- function(backend) {
  if (identical(backend, "base")) return(list(available = TRUE))
  tryCatch(
    cudaverse::cuda_diagnostics()$backend_diagnostics[[backend]],
    error = function(error) list(
      available = FALSE,
      detection_error = conditionMessage(error)
    )
  )
}

backend_synchronize <- function(backend) {
  if (identical(backend, "native")) {
    factory$synchronize()
  } else if (identical(backend, "torch")) {
    torch::cuda_synchronize()
  }
  invisible(NULL)
}

pipeline_once <- function(backend, source, n_components) {
  requested_backends <- if (identical(backend, "native")) NULL else backend
  old_options <- options(cudaverse.cuda_backends = requested_backends)
  on.exit(options(old_options), add = TRUE)
  device <- if (identical(backend, "base")) {
    "cpu"
  } else if (identical(backend, "native")) {
    "auto"
  } else {
    "cuda"
  }
  start <- proc.time()[["elapsed"]]
  sparse <- cudaverse::cuda_sparse(
    source, format = "csr", device = device
  )
  backend_synchronize(backend)
  transfer_end <- proc.time()[["elapsed"]]
  normalized <- cudaverse::sparse_normalize(
    sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
  )
  backend_synchronize(backend)
  normalization_end <- proc.time()[["elapsed"]]
  fit <- cudaverse::cuda_pca(
    normalized,
    n_components = n_components,
    center = TRUE,
    scale. = FALSE,
    device = device
  )
  backend_synchronize(backend)
  pca_end <- proc.time()[["elapsed"]]
  neighbors <- cudaverse::cuda_knn(
    fit$x,
    k = k,
    metric = "euclidean",
    device = device,
    batch_size = batch_size
  )
  backend_synchronize(backend)
  end <- proc.time()[["elapsed"]]
  list(
    value = list(
      sparse = sparse,
      normalized = normalized,
      pca = fit,
      knn = neighbors
    ),
    seconds = c(
      transfer = transfer_end - start,
      normalization = normalization_end - transfer_end,
      pca = pca_end - normalization_end,
      knn = end - pca_end,
      full_pipeline = end - start
    )
  )
}

benchmark_backend <- function(backend, source, n_components) {

  cold <- pipeline_once(backend, source, n_components)
  cold_seconds <- cold$seconds
  cold$value <- NULL
  rm(cold)
  invisible(gc(FALSE))
  for (index in seq_len(warmup_runs)) {
    warm <- pipeline_once(backend, source, n_components)
    warm$value <- NULL
    rm(warm)
    invisible(gc(FALSE))
  }

  measurements <- matrix(
    NA_real_, nrow = timed_runs, ncol = 5L,
    dimnames = list(
      NULL,
      c("transfer", "normalization", "pca", "knn", "full_pipeline")
    )
  )
  last <- NULL
  for (index in seq_len(timed_runs)) {
    timed <- pipeline_once(backend, source, n_components)
    measurements[index, ] <- timed$seconds
    if (index == timed_runs) {
      last <- timed$value
    } else {
      timed$value <- NULL
    }
    rm(timed)
    if (index != timed_runs) invisible(gc(FALSE))
  }

  list(
    cold_seconds = cold_seconds,
    measurements = measurements,
    last = last
  )
}

for (case_name in names(cases)) {
  case <- cases[[case_name]]
  if (k >= case$rows) stop("`k` must be smaller than every row count.")
  n_components <- min(components, case$columns, case$rows - 1L)
  source <- make_sparse_input(
    case$rows,
    case$columns,
    case$density,
    seed = 20260809L + case$rows + case$columns
  )
  observed_density <- Matrix::nnzero(source) / prod(dim(source))
  report$benchmarks[[case_name]] <- list()
  reference <- NULL

  for (backend in backends) {
    message("Benchmarking ", case_name, " with ", backend, " backend")
    diagnostics <- backend_diagnostics(backend)
    if (!isTRUE(diagnostics$available)) {
      report$benchmarks[[case_name]][[backend]] <- list(
        status = "skipped",
        reason = diagnostics$detection_error
      )
      write_report()
      next
    }

    result <- benchmark_backend(backend, source, n_components)
    if (identical(backend, "base")) {
      reference <- reference_payload(result$last)
    }
    if (is.null(reference)) {
      stop("The base backend must run before GPU backends for each case.")
    }
    validation <- validation_payload(result$last, reference)
    provenance <- list(
      normalized = provenance_payload(result$last$normalized),
      pca = provenance_payload(result$last$pca),
      knn = provenance_payload(result$last$knn),
      sparse_storage_device_resident = identical(
        result$last$normalized$device, "cuda"
      ) && typeof(result$last$normalized$storage) == "externalptr",
      pca_scores_device_resident = local({
        score_state <- attr(
          result$last$pca$x,
          "cudaverse_native_state",
          exact = TRUE
        )
        is.list(score_state) && typeof(score_state$storage) == "externalptr"
      })
    )

    cold_seconds <- result$cold_seconds
    measurements <- result$measurements

    result$last <- NULL
    invisible(gc(FALSE))
    backend_synchronize(backend)

    tracker_before <- cudaverse:::.native_memory_tracker(reset = TRUE)
    device_before <- cudaverse:::.native_memory_info()
    memory_run <- pipeline_once(backend, source, n_components)
    tracker_during <- cudaverse:::.native_memory_tracker()
    device_during <- cudaverse:::.native_memory_info()
    torch_memory <- if (identical(backend, "torch")) {
      torch::cuda_memory_stats()
    } else {
      NULL
    }
    memory_run$value <- NULL
    rm(memory_run)
    rm(result)
    invisible(gc(FALSE))
    factory$synchronize()
    if (identical(backend, "torch")) torch::cuda_synchronize()
    tracker_after <- cudaverse:::.native_memory_tracker()
    device_after <- cudaverse:::.native_memory_info()

    memory <- list(
      native_tracker_current_before_bytes = tracker_before$current,
      native_tracker_peak_delta_bytes = tracker_during$peak -
        tracker_before$current,
      native_tracker_current_with_result_delta_bytes =
        tracker_during$current - tracker_before$current,
      native_tracker_post_cleanup_difference_bytes =
        tracker_after$current - tracker_before$current,
      whole_device_used_before_bytes = device_before$used,
      whole_device_used_with_result_bytes = device_during$used,
      whole_device_used_after_cleanup_bytes = device_after$used,
      whole_device_post_cleanup_absolute_difference_bytes =
        abs(device_after$used - device_before$used),
      backend_allocator_peak_bytes = if (identical(backend, "native")) {
        tracker_during$peak - tracker_before$current
      } else if (identical(backend, "torch")) {
        unname(torch_memory$allocated_bytes$all$peak)
      } else {
        0
      },
      backend_allocator_reserved_peak_bytes = if (identical(backend, "torch")) {
        unname(torch_memory$reserved_bytes$all$peak)
      } else {
        NA_real_
      },
      backend_allocator_peak_source = if (identical(backend, "native")) {
        "cudaverse native operation-owned allocation tracker"
      } else if (identical(backend, "torch")) {
        "torch CUDA allocator session high-water mark"
      } else {
        "not applicable (CPU)"
      }
    )

    report$benchmarks[[case_name]][[backend]] <- list(
      status = "complete",
      rows = case$rows,
      columns = case$columns,
      requested_density = case$density,
      observed_density = observed_density,
      nnz = Matrix::nnzero(source),
      n_components = n_components,
      cold_seconds = as.list(cold_seconds),
      transfer = summary_times(measurements[, "transfer"]),
      normalization = summary_times(measurements[, "normalization"]),
      pca = summary_times(measurements[, "pca"]),
      knn = summary_times(measurements[, "knn"]),
      full_pipeline = summary_times(
        measurements[, "full_pipeline"]
      ),
      validation = validation,
      provenance = provenance,
      memory = memory
    )
    write_report()
  }

  rm(source, reference)
  invisible(gc(FALSE))
}

message("Running Phase 4 RTX lifecycle and recovery validation")
options(cudaverse.cuda_backends = NULL)

surface_values <- matrix(seq_len(12) / 7, 4L, 3L)
surface_other <- matrix(seq_len(12) / 11 + 1, 4L, 3L)
surface_right <- matrix(seq_len(6) / 13, 3L, 2L)
surface_vector <- c(2, 3, 4)
surface_results <- list()
max_abs_error <- function(actual, expected) {
  max(abs(as.vector(cudaverse::to_cpu(actual)) - as.vector(expected)))
}
for (dtype in c("float32", "float64")) {
  x <- cudaverse::cuda_tensor(
    surface_values, dtype = dtype, device = "auto"
  )
  y <- cudaverse::cuda_tensor(
    surface_other, dtype = dtype, device = "auto"
  )
  right <- cudaverse::cuda_tensor(
    surface_right, dtype = dtype, device = "auto"
  )
  vector <- cudaverse::cuda_tensor(
    surface_vector, dtype = dtype, device = "auto"
  )
  outputs <- list(
    add = x + y,
    subtract = x - y,
    multiply = x * y,
    divide = x / y,
    power = x^y,
    broadcast = x + vector,
    reshape = cudaverse::tensor_reshape(x, c(3L, 4L)),
    transpose = t(x),
    matmul = cudaverse::tensor_matmul(x, right),
    reduction = cudaverse::tensor_sum(x)
  )
  expected <- list(
    add = surface_values + surface_other,
    subtract = surface_values - surface_other,
    multiply = surface_values * surface_other,
    divide = surface_values / surface_other,
    power = surface_values^surface_other,
    broadcast = sweep(surface_values, 2L, surface_vector, "+"),
    reshape = array(surface_values, dim = c(3L, 4L)),
    transpose = t(surface_values),
    matmul = surface_values %*% surface_right,
    reduction = sum(surface_values)
  )
  errors <- vapply(
    names(outputs),
    function(operation) max_abs_error(outputs[[operation]], expected[[operation]]),
    numeric(1)
  )
  native_devices <- vapply(
    outputs,
    function(value) identical(
      cudaverse::tensor_device(value),
      c(device = "cuda", backend = "native")
    ),
    logical(1)
  )
  tolerance <- if (identical(dtype, "float32")) 1e-5 else 1e-10
  surface_results[[dtype]] <- list(
    operation_max_absolute_error = as.list(errors),
    tolerance = tolerance,
    all_outputs_native = all(native_devices),
    passed = all(errors <= tolerance) && all(native_devices)
  )
  rm(x, y, right, vector, outputs)
  invisible(gc(FALSE))
}
dtype_surface <- list(
  operations = names(expected),
  dtypes = surface_results,
  passed = all(vapply(surface_results, `[[`, logical(1), "passed"))
)

factory$synchronize()
invisible(gc())
dense_device_before <- cudaverse:::.native_memory_info()$used
dense_tracker_before <-
  cudaverse:::.native_memory_tracker(reset = TRUE)$current
dense_values <- matrix(seq_len(64) / 13, 8L, 8L)
for (iteration in seq_len(1000L)) {
  dense_storage <- factory$from_host(
    dense_values, "float64", dim(dense_values)
  )
  roundtrip <- factory$to_host(dense_storage)
  if (!isTRUE(all.equal(roundtrip, as.vector(dense_values), tolerance = 0))) {
    stop("Dense lifecycle transfer parity failed at cycle ", iteration, ".")
  }
  factory$release(dense_storage)
}
factory$synchronize()
invisible(gc())
dense_device_after <- cudaverse:::.native_memory_info()$used
dense_tracker_after <- cudaverse:::.native_memory_tracker()
dense_lifecycle <- list(
  cycles = 1000L,
  whole_device_absolute_difference_bytes = abs(
    dense_device_after - dense_device_before
  ),
  tracked_current_difference_bytes =
    dense_tracker_after$current - dense_tracker_before,
  tracked_peak_delta_bytes = dense_tracker_after$peak - dense_tracker_before
)
dense_lifecycle$passed <-
  dense_lifecycle$whole_device_absolute_difference_bytes <= 1024^2 &&
  identical(dense_lifecycle$tracked_current_difference_bytes, 0)

factory$synchronize()
invisible(gc())
lifecycle_device_before <- cudaverse:::.native_memory_info()$used
lifecycle_tracker_before <-
  cudaverse:::.native_memory_tracker(reset = TRUE)$current
i <- rep(seq_len(16L), each = 3L)
j <- rep(c(1L, 4L, 8L), 16L)
values <- rep(c(1, 2, 3), 16L)
for (iteration in seq_len(1000L)) {
  storage <- factory$sparse_from_coo(i, j, values, c(16L, 8L), "csr")
  normalized <- factory$sparse_normalize(storage, 0L, 100, TRUE)
  factory$sparse_release(normalized$storage)
  factory$sparse_release(storage)
}
factory$synchronize()
invisible(gc())
lifecycle_device_after <- cudaverse:::.native_memory_info()$used
lifecycle_tracker_after <- cudaverse:::.native_memory_tracker()
lifecycle <- list(
  cycles = 1000L,
  whole_device_absolute_difference_bytes = abs(
    lifecycle_device_after - lifecycle_device_before
  ),
  tracked_current_difference_bytes =
    lifecycle_tracker_after$current - lifecycle_tracker_before,
  tracked_peak_delta_bytes =
    lifecycle_tracker_after$peak - lifecycle_tracker_before
)
lifecycle$passed <- lifecycle$whole_device_absolute_difference_bytes <= 1024^2 &&
  identical(lifecycle$tracked_current_difference_bytes, 0)

shared_source <- Matrix::sparseMatrix(
  i = c(1L, 1L, 2L, 4L),
  j = c(1L, 3L, 2L, 3L),
  x = c(2, 1, 4, 5),
  dims = c(4L, 3L)
)
shared_csr <- cudaverse::cuda_sparse(
  shared_source, format = "csr", device = "cuda"
)
shared_coo <- cudaverse::as_coo(shared_csr)
factory$sparse_release(shared_csr$storage)
shared_roundtrip <- isTRUE(all.equal(
  as.matrix(cudaverse::to_dgCMatrix(shared_coo)),
  as.matrix(shared_source),
  tolerance = 0
))
shared_final_release <- factory$sparse_release(shared_coo$storage)
shared_released_error <- inherits(tryCatch(
  factory$sparse_to_host(shared_coo$storage), error = identity
), "error")
shared_ownership <- list(
  roundtrip_after_first_owner_release = shared_roundtrip,
  final_release_succeeded = isTRUE(shared_final_release),
  use_after_final_release_errors = shared_released_error,
  passed = shared_roundtrip && isTRUE(shared_final_release) &&
    shared_released_error
)
rm(shared_csr, shared_coo)
invisible(gc(FALSE))

error_source <- Matrix::Diagonal(4L, x = 1:4)
error_sparse <- cudaverse::cuda_sparse(error_source, device = "cuda")
broken <- error_sparse
broken$shape <- c(4L, 3L)
condition <- tryCatch(
  cudaverse::sparse_matmul_dense(broken, matrix(1, 3, 2)),
  error = identity
)
backend_reusable <- isTRUE(all.equal(
  as.numeric(cudaverse::sparse_row_sums(error_sparse)),
  as.numeric(1:4),
  tolerance = 0
))
structured_error <- list(
  classes = class(condition),
  backend = condition$backend,
  operation = condition$operation,
  message = conditionMessage(condition),
  backend_reusable = backend_reusable
)
structured_error$passed <- inherits(condition, "cudaverse_native_error") &&
  identical(condition$backend, "native") &&
  identical(condition$operation, "sparse_matmul_dense") &&
  backend_reusable

factory$synchronize()
invisible(gc(FALSE))
injected_before <- cudaverse:::.native_memory_tracker(reset = TRUE)$current
injected <- tryCatch(
  cudaverse:::.backend_call(
    "native", "test_inject_cuda_error", 4096L
  ),
  error = identity
)
factory$synchronize()
invisible(gc(FALSE))
injected_after <- cudaverse:::.native_memory_tracker()
injected_error <- list(
  classes = class(injected),
  backend = injected$backend,
  operation = injected$operation,
  message = conditionMessage(injected),
  tracked_current_difference_bytes = injected_after$current - injected_before
)
injected_error$passed <- inherits(injected, "cudaverse_native_error") &&
  identical(injected$backend, "native") &&
  identical(injected$operation, "test_inject_cuda_error") &&
  identical(injected_error$tracked_current_difference_bytes, 0)

set.seed(20260809L)
interrupt_source <- abs(Matrix::rsparsematrix(512, 256, density = 0.05))
interrupt_sparse <- cudaverse::cuda_sparse(
  interrupt_source, device = "cuda"
)
interrupt_dense <- matrix(stats::rnorm(256 * 64), 256, 64)
factory$synchronize()
invisible(gc(FALSE))
interrupt_before <- cudaverse:::.native_memory_tracker(reset = TRUE)$current
interrupt_condition <- tryCatch({
  setTimeLimit(elapsed = 0.01, transient = TRUE)
  repeat {
    interrupt_product <- cudaverse::sparse_matmul_dense(
      interrupt_sparse, interrupt_dense
    )
    cudaverse::to_cpu(interrupt_product)
  }
}, error = identity, finally = setTimeLimit(
  cpu = Inf, elapsed = Inf, transient = FALSE
))
if (exists("interrupt_product", inherits = FALSE)) rm(interrupt_product)
invisible(gc(FALSE))
factory$synchronize()
interrupt_after <- cudaverse:::.native_memory_tracker()$current
interrupt_reusable <- isTRUE(all.equal(
  as.numeric(cudaverse::sparse_row_sums(interrupt_sparse)),
  as.numeric(Matrix::rowSums(interrupt_source)),
  tolerance = 1e-10
))
interruption <- list(
  condition_message = conditionMessage(interrupt_condition),
  interrupted = inherits(interrupt_condition, "error") && grepl(
    "time limit|elapsed",
    conditionMessage(interrupt_condition),
    ignore.case = TRUE
  ),
  tracked_current_difference_bytes = interrupt_after - interrupt_before,
  backend_reusable = interrupt_reusable
)
interruption$passed <- isTRUE(interruption$interrupted) &&
  identical(interruption$tracked_current_difference_bytes, 0) &&
  interrupt_reusable

resident_source <- make_sparse_input(256L, 48L, 0.08, 20260810L)
resident_sparse <- cudaverse::cuda_sparse(resident_source, device = "cuda")
resident_normalized <- cudaverse::sparse_normalize(
  resident_sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
)
resident_pca <- cudaverse::cuda_pca(
  resident_normalized,
  n_components = 12L,
  center = TRUE,
  scale. = FALSE,
  device = "cuda"
)
resident_knn <- cudaverse::cuda_knn(
  resident_pca$x,
  k = 15L,
  device = "cuda",
  batch_size = 64L
)
resident_normalized_provenance <- provenance_payload(resident_normalized)
resident_pca_provenance <- provenance_payload(resident_pca)
resident_knn_provenance <- provenance_payload(resident_knn)
score_state <- attr(
  resident_pca$x, "cudaverse_native_state", exact = TRUE
)
resident_pipeline <- list(
  provenance_schema = c(
    normalized = resident_normalized_provenance$schema,
    pca = resident_pca_provenance$schema,
    knn = resident_knn_provenance$schema
  ),
  normalized_stages = resident_normalized_provenance$stages$stage,
  pca_stages = resident_pca_provenance$stages$stage,
  knn_stages = resident_knn_provenance$stages$stage,
  normalized_storage_external_pointer =
    typeof(resident_normalized$storage) == "externalptr",
  pca_scores_storage_external_pointer =
    is.list(score_state) && typeof(score_state$storage) == "externalptr",
  normalized_compute_device = resident_normalized_provenance$compute_device,
  pca_compute_device = resident_pca_provenance$compute_device,
  knn_compute_device = resident_knn_provenance$compute_device
)
resident_pipeline$passed <- all(
  resident_pipeline$provenance_schema == "cudaverse-stage/1"
) && isTRUE(resident_pipeline$normalized_storage_external_pointer) &&
  isTRUE(resident_pipeline$pca_scores_storage_external_pointer) &&
  all(c(
    "normalization", "sparse_to_dense", "preprocessing",
    "decomposition", "scores_resident"
  ) %in% resident_pipeline$pca_stages) &&
  all(c("distance", "neighbor_selection") %in%
        resident_pipeline$knn_stages) &&
  identical(resident_pipeline$normalized_compute_device, "cuda") &&
  identical(resident_pipeline$pca_compute_device, "cuda") &&
  identical(resident_pipeline$knn_compute_device, "cuda")

tie_values <- rbind(
  c(0, 0), c(1, 0), c(-1, 0), c(0, 1), c(0, -1), c(1, 0)
)
tie_sparse <- cudaverse::cuda_sparse(
  Matrix::Matrix(tie_values, sparse = TRUE), device = "cuda"
)
tie_native <- cudaverse::cuda_knn(
  tie_sparse, k = 4L, device = "cuda", batch_size = 2L
)
tie_cpu <- cudaverse::cuda_knn(
  tie_sparse, k = 4L, device = "cpu", batch_size = 2L
)
stable_ties <- list(
  native_index = unname(tie_native$index),
  cpu_index = unname(tie_cpu$index),
  indices_identical = identical(
    unname(tie_native$index), unname(tie_cpu$index)
  ),
  distance_max_absolute_error = max(abs(
    unname(tie_native$distance) - unname(tie_cpu$distance)
  )),
  rule = "equal distances are ordered by original one-based row index"
)
stable_ties$passed <- stable_ties$indices_identical &&
  stable_ties$distance_max_absolute_error <= 1e-10

report$hardware_validation <- list(
  schema = "cudaverse-native-consolidation-validation/1",
  auto_selection = auto_selection,
  dtype_surface = dtype_surface,
  dense_lifecycle = dense_lifecycle,
  lifecycle = lifecycle,
  shared_ownership = shared_ownership,
  structured_error = structured_error,
  injected_cuda_error = injected_error,
  interruption = interruption,
  resident_pipeline = resident_pipeline,
  stable_ties = stable_ties
)

benchmark_pass <- all(vapply(
  report$benchmarks,
  function(case) all(vapply(
    case,
    function(result) identical(result$status, "complete") &&
      isTRUE(result$validation$passed),
    logical(1)
  )),
  logical(1)
))
validation_pass <- all(vapply(
  report$hardware_validation[-1L],
  function(value) isTRUE(value$passed),
  logical(1)
))
json_number <- function(value) {
  as.numeric(unlist(value, recursive = TRUE, use.names = FALSE)[[1L]])
}
regression_cases <- list()
for (case_name in names(report$benchmarks)) {
  candidate <- report$benchmarks[[case_name]]$native
  previous <- baseline$benchmarks[[case_name]]$native
  if (is.null(previous)) {
    stop("The Phase 3 baseline is missing native case `", case_name, "`.")
  }
  baseline_median <- json_number(previous$full_pipeline$median_seconds)
  baseline_p95 <- json_number(previous$full_pipeline$p95_seconds)
  baseline_peak <- json_number(
    previous$memory$backend_allocator_peak_bytes
  )
  candidate_median <- candidate$full_pipeline$median_seconds
  candidate_p95 <- candidate$full_pipeline$p95_seconds
  candidate_peak <- candidate$memory$backend_allocator_peak_bytes
  median_limit <- max(baseline_median * 1.20, baseline_median + 0.02)
  p95_limit <- max(baseline_p95 * 1.25, baseline_p95 + 0.03)
  peak_limit <- max(baseline_peak * 1.10, baseline_peak + 1024^2)
  regression_cases[[case_name]] <- list(
    baseline_median_seconds = baseline_median,
    candidate_median_seconds = candidate_median,
    median_limit_seconds = median_limit,
    median_passed = candidate_median <= median_limit,
    baseline_p95_seconds = baseline_p95,
    candidate_p95_seconds = candidate_p95,
    p95_limit_seconds = p95_limit,
    p95_passed = candidate_p95 <= p95_limit,
    baseline_peak_bytes = baseline_peak,
    candidate_peak_bytes = candidate_peak,
    peak_limit_bytes = peak_limit,
    peak_passed = candidate_peak <= peak_limit
  )
  regression_cases[[case_name]]$passed <- with(
    regression_cases[[case_name]],
    median_passed && p95_passed && peak_passed
  )
}
combined_baseline_size <-
  json_number(baseline$installed_size_bytes$cudaverse) +
  json_number(baseline$installed_size_bytes$cudaverseCUDA)
candidate_size <- report$installed_size_bytes$cudaverse
size_limit <- max(
  combined_baseline_size * 1.25,
  combined_baseline_size + 128 * 1024
)
size_regression <- list(cudaverse = list(
  baseline_bytes = combined_baseline_size,
  baseline_definition = "legacy Phase 3 installed footprint baseline",
  candidate_bytes = candidate_size,
  limit_bytes = size_limit,
  passed = candidate_size <= size_limit
))
report$benchmark_regression <- list(
  schema = "cudaverse-native-consolidation-regression/1",
  role = "advisory",
  release_gate = FALSE,
  authority = "cudaverse-benchmark/1 full profile",
  baseline_report = basename(baseline_path),
  baseline_schema = baseline$schema,
  baseline_sha256 = baseline_sha256,
  policy = list(
    median = "no more than 20% or 0.02 seconds above Phase 3",
    p95 = "no more than 25% or 0.03 seconds above Phase 3",
    native_peak_vram = "no more than 10% or 1 MiB above Phase 3",
    installed_size = "no more than 25% or 128 KiB above Phase 3"
  ),
  cases = regression_cases,
  installed_size = size_regression
)
report$benchmark_regression$passed <-
  isTRUE(baseline$overall_pass) &&
  all(vapply(regression_cases, `[[`, logical(1), "passed")) &&
  all(vapply(size_regression, `[[`, logical(1), "passed"))
report$overall_pass <- benchmark_pass && validation_pass
write_report()
options(original_backend_options)

if (!isTRUE(report$overall_pass)) {
  stop("Consolidation report completed, but one or more gates failed.")
}
message("Wrote passing consolidation report to ", normalizePath(output))

