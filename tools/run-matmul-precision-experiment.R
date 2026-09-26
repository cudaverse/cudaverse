# Native matmul comparison with common float32-rounded host inputs.
# Set CUDAVERSE_MATMUL_PRECISION_OUTPUT to a NEW JSON path. Set
# CUDAVERSE_MATMUL_PRECISION_PROFILE=diagnostic (default) or full.
# Diagnostic: 64/256/1024 + exact controls, 1 warmup and 3 measured samples;
# timings on a shared GPU are diagnostic and cannot support speed claims.
# Full: also 4096, 5 warmups and 10 samples, clean source and mandatory idle
# GPU checks before/after every cold, warmup and retained timing sample.
# Select the intended installed package/library before process startup.
run_matmul_precision_experiment <- function() {
  output <- Sys.getenv("CUDAVERSE_MATMUL_PRECISION_OUTPUT", "")
  if (!nzchar(output) || file.exists(output)) {
    stop("CUDAVERSE_MATMUL_PRECISION_OUTPUT must name a new JSON file.", call. = FALSE)
  }
  output <- file.path(normalizePath(dirname(output), mustWork = TRUE), basename(output))
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required to preserve the report.", call. = FALSE)
  }
  profile <- Sys.getenv("CUDAVERSE_MATMUL_PRECISION_PROFILE", "diagnostic")
  report <- list(
    schema = "cudaverse-matmul-precision-experiment/1", profile = profile,
    started_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    passed = FALSE, complete = FALSE, retained_performance_eligible = FALSE,
    cases = list()
  )
  command_result <- function(command, args) {
    tryCatch({
      value <- suppressWarnings(system2(command, args, stdout = TRUE, stderr = TRUE))
      status <- attr(value, "status", exact = TRUE)
      list(status = if (is.null(status)) 0L else as.integer(status),
           output = as.character(value))
    }, error = function(e) list(error = conditionMessage(e), status = NA_integer_))
  }
  require_check <- function(value, message) {
    if (!isTRUE(value)) stop(message, call. = FALSE)
    invisible(TRUE)
  }
  fail_report <- function(e) {
    report$runtime_error <<- conditionMessage(e)
    report$passed <<- FALSE
    report$complete <<- FALSE
  }
  tryCatch({
    require_check(profile %in% c("diagnostic", "full"),
                  "The precision profile must be diagnostic or full.")
    for (package in c("digest", "cudaverse")) {
      require_check(requireNamespace(package, quietly = TRUE),
                    paste("Required package is unavailable:", package))
    }
    helper_paths <- file.path("tools", c(
      "benchmark-validation.R", "matmul-precision-validation.R",
      "benchmark-timing.R", "benchmark-gpu-guard.R"))
    identity_helper <- file.path("tools", "benchmark-source-identity.R")
    if (file.exists(identity_helper)) helper_paths <- c(helper_paths, identity_helper)
    for (path in helper_paths) sys.source(path, envir = environment())
    require_check(is.function(benchmark_float32_matmul_validation),
                  "The standard FP32 validation helper is unavailable.")
    sha <- function(path) digest::digest(file = path, algo = "sha256")
    object_sha <- function(x) digest::digest(x, algo = "sha256")
    # Hash raw double encodings to distinguish signed zero as well as values.
    value_sha <- function(x) digest::digest(
      writeBin(as.numeric(x), raw(), size = 8L), algo = "sha256", serialize = FALSE)
    full <- identical(profile, "full")
    warmups <- if (full) 5L else 1L
    timed_runs <- if (full) 10L else 3L
    report$timing_contract <- list(
      warmups = warmups, timed_runs = timed_runs,
      synchronized = TRUE, clock = "elapsed wall clock via Sys.time",
      resident_compute = "resident tensors -> public tensor_matmul -> synchronized resident tensor",
      host_boundary = "host float32-rounded matrices -> tensors -> public tensor_matmul -> synchronized host matrix",
      validation_and_gc = "outside timed intervals; allocations and API overhead inside intervals",
      cold_scope = "first scope sample after backend initialization and correctness probes; not a fresh-process CUDA cold start",
      idle_guard_required = full,
      shared_gpu_warning = if (!full) paste(
        "Diagnostic profile: GPU sharing is allowed. These durations are",
        "not retained performance evidence and support no speed claim."
      ) else NULL,
      mode_order = c("standard_float32", "tf32_permitted_float32", "standard_float64")
    )
    report$policies <- list(
      standard_float32 = benchmark_float32_matmul_policy(),
      tf32_permitted_float32 = matmul_precision_tf32_policy(),
      standard_float64 = list(rtol = 1e-8, atol = 1e-10),
      exact_controls = "All modes must exactly match the reference on small integer identity and odd rectangular controls"
    )
    source_commit <- command_result("git", c("rev-parse", "HEAD"))
    source_status <- command_result("git", c("status", "--porcelain", "--untracked-files=all"))
    script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    require_check(length(script_arg) == 1L, "Invoke this driver as an Rscript file.")
    script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
    source_files <- unique(c(script_path, helper_paths, "DESCRIPTION", "R/tensor.R",
                             "R/backend-native.R", "src/native_backend.cpp",
                             "inst/kernels/cudaverse_dense_kernels.ptx"))
    report$source <- list(
      directory = normalizePath(getwd(), mustWork = TRUE),
      commit = source_commit, status = source_status,
      clean = identical(source_status$status, 0L) && !length(source_status$output),
      file_sha256 = as.list(setNames(vapply(source_files, sha, character(1)), source_files))
    )
    require_check(identical(source_commit$status, 0L), "Could not identify the source revision.")
    if (full) require_check(report$source$clean, "The full retained profile requires a clean source checkout.")
    package_path <- normalizePath(find.package("cudaverse"), mustWork = TRUE)
    dll_path <- normalizePath(getLoadedDLLs()[["cudaverse"]][["path"]], mustWork = TRUE)
    installed_ptx <- file.path(package_path, "kernels", "cudaverse_dense_kernels.ptx")
    report$installed <- list(
      package = package_path, version = as.character(packageVersion("cudaverse")),
      dll = dll_path, dll_sha256 = sha(dll_path),
      description_sha256 = sha(file.path(package_path, "DESCRIPTION")),
      ptx_sha256 = sha(installed_ptx), library_paths = .libPaths()
    )
    report$installed$source_identity <- if (exists(
      "benchmark_installed_source_identity", envir = environment(), inherits = FALSE
    )) tryCatch(
      benchmark_installed_source_identity(source_root = ".", package = "cudaverse"),
      error = function(e) list(verified = FALSE, error = conditionMessage(e))) else
        list(verified = FALSE, reason = "source identity helper is unavailable")
    if (full) require_check(isTRUE(report$installed$source_identity$verified),
                            "The full profile requires a verified installed build of this source; rebuild the package from this checkout.")
    require_check(identical(report$installed$version,
                            unname(read.dcf("DESCRIPTION", fields = "Version")[[1L]])),
                  "Installed and source package versions differ.")
    require_check(identical(report$installed$ptx_sha256,
                            sha("inst/kernels/cudaverse_dense_kernels.ptx")),
                  "Installed and source PTX artifacts differ.")
    require_check("precision" %in% names(formals(cudaverse::tensor_matmul)),
                  "Installed tensor_matmul lacks the per-call precision API.")
    environment_names <- c(
      "CUDAVERSE_MATMUL_PRECISION_PROFILE", "CUDAVERSE_NATIVE_TESTS",
      "CUDAVERSE_CUBLAS_PATH", "CUDAVERSE_CUSOLVER_PATH", "CUDA_VISIBLE_DEVICES",
      "NVIDIA_TF32_OVERRIDE", "R_LIBS_USER", "R_LIBS", "OMP_NUM_THREADS",
      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
    )
    report$environment <- list(
      variables = as.list(Sys.getenv(environment_names, unset = "<unset>")),
      pid = Sys.getpid(), torch_installed = length(find.package("torch", quiet = TRUE)) > 0L,
      hardware = command_result("nvidia-smi", c(
        "--query-gpu=name,uuid,driver_version,compute_cap,memory.total", "--format=csv,noheader")),
      compute_processes_at_start = command_result("nvidia-smi", c(
        "--query-compute-apps=pid,process_name", "--format=csv,noheader"))
    )
    library_identity <- function(variable) {
      path <- Sys.getenv(variable, "")
      present <- nzchar(path) && file.exists(path)
      list(configured = path, exists = present,
           normalized_path = if (present) normalizePath(path, mustWork = TRUE) else NULL,
           bytes = if (present) unname(file.info(path)$size) else NULL)
    }
    report$environment$cuda_libraries <- list(
      cublas = library_identity("CUDAVERSE_CUBLAS_PATH"),
      cusolver = library_identity("CUDAVERSE_CUSOLVER_PATH"))
    require_check(!identical(Sys.getenv("NVIDIA_TF32_OVERRIDE"), "0"),
                  "TF32 comparison requires a new process without NVIDIA_TF32_OVERRIDE=0.")
    benchmark_validate_idle_gpu_requirement(if (full) "full" else "smoke", "native", full)
    if (full) benchmark_assert_idle_gpu("precision experiment startup")
    old_options <- options(cudaverse.cuda_backends = "native")
    on.exit(options(old_options), add = TRUE)
    report$diagnostics <- cudaverse::cuda_diagnostics()
    require_check(identical(report$diagnostics$selected_backend, "native"),
                  "The selected backend is not native CUDA.")
    factory <- cudaverse:::.native_backend_factory()
    factory$synchronize()
    invisible(gc())
    report$initial_memory <- cudaverse:::.native_memory_tracker()
    RNGkind("Mersenne-Twister", "Inversion", "Rejection")
    report$rng_kind <- RNGkind()
    fixture <- function(id, left, right, exact = FALSE, seed = NULL, labels = TRUE) {
      left <- benchmark_float32_input(left)
      right <- benchmark_float32_input(right)
      if (labels) {
        dimnames(left) <- list(paste0("row", seq_len(nrow(left))),
                               paste0("inner", seq_len(ncol(left))))
        dimnames(right) <- list(colnames(left), paste0("column", seq_len(ncol(right))))
      }
      list(id = id, left = left, right = right, reference = left %*% right,
           exact = exact, seed = seed)
    }
    make_fixture <- function(id) {
      if (identical(id, "control-identity")) {
        return(fixture(id, matrix(as.double(seq_len(77L) %% 11L - 5L), 7L, 11L),
                       diag(11L), exact = TRUE))
      }
      if (identical(id, "control-odd-rectangular")) {
        return(fixture(id, matrix(as.double(seq_len(15L) %% 7L - 3L), 3L, 5L),
                       matrix(as.double(seq_len(35L) %% 9L - 4L), 5L, 7L), exact = TRUE))
      }
      size <- as.integer(sub("^matmul-float32-", "", id))
      seed <- 20260810L + sum(utf8ToInt(id))
      set.seed(seed)
      fixture(id, matrix(rnorm(size * size), size, size),
               matrix(rnorm(size * size), size, size), seed = seed,
               labels = size != 64L)
    }
    case_ids <- c("control-identity", "control-odd-rectangular",
                  paste0("matmul-float32-", if (full) c(64L, 256L, 1024L, 4096L) else c(64L, 256L, 1024L)))
    report$expected_case_ids <- case_ids
    modes <- list(
      standard_float32 = list(dtype = "float32", precision = "standard"),
      tf32_permitted_float32 = list(dtype = "float32", precision = "tf32"),
      standard_float64 = list(dtype = "float64", precision = "standard")
    )
    inspect_tensor <- function(x, mode, fixture) {
      provenance <- cudaverse::cuda_provenance(x)
      expected_reason <- if (mode$precision == "tf32") "matmul_tf32_permitted" else "matmul_standard"
      checks <- list(
        dtype = identical(x$dtype, mode$dtype),
        shape = identical(cudaverse::tensor_shape(x), dim(fixture$reference)),
        dimnames = identical(dimnames(x), dimnames(fixture$reference)),
        native_device = identical(cudaverse::tensor_device(x), c(device = "cuda", backend = "native")),
        provenance = identical(provenance$stage, "matrix_multiply") &&
          identical(provenance$selection_reason, expected_reason) &&
          all(provenance$device == "cuda") && all(provenance$backend == "native") &&
          all(provenance$output_device == "cuda") && all(!provenance$fallback)
      )
      list(checks = checks, passed = all(unlist(checks)), dtype = x$dtype,
           shape = cudaverse::tensor_shape(x), provenance = as.data.frame(provenance))
    }
    validate_mode <- function(actual, mode, fixture) {
      if (mode$dtype == "float64") {
        return(benchmark_numeric_validation(actual, fixture$reference, 1e-8, 1e-10))
      }
      if (mode$precision == "tf32") {
        return(matmul_precision_tf32_validation(actual, fixture$reference,
                                                fixture$left, fixture$right))
      }
      benchmark_float32_matmul_validation(actual, fixture$reference,
                                           fixture$left, fixture$right)
    }
    # The wrapper receives host-only summaries. Its cleanup check therefore
    # covers tensors created anywhere inside the completed or failed call.
    with_memory_check <- function(run) {
      factory$synchronize()
      invisible(gc())
      before <- cudaverse:::.native_memory_tracker(reset = TRUE)
      value <- tryCatch(run(), error = function(e) list(passed = FALSE, error = conditionMessage(e)),
                        interrupt = function(e) list(passed = FALSE, error = "interrupted"))
      after <- tryCatch({
        factory$synchronize()
        invisible(gc())
        factory$synchronize()
        cudaverse:::.native_memory_tracker()
      }, error = function(e) list(error = conditionMessage(e)))
      clean <- !is.null(after$current) && identical(as.numeric(after$current), as.numeric(before$current))
      value$memory <- list(
        scope = "cudaverse-owned native allocations; excludes CUDA libraries and external processes",
        reset_baseline = before, after_cleanup = after,
        peak_above_baseline_bytes = if (!is.null(after$peak)) after$peak - before$current else NULL,
        cleanup_difference_bytes = if (!is.null(after$current)) after$current - before$current else NULL,
        passed = clean)
      value$passed <- isTRUE(value$passed) && clean
      value
    }
    correctness_probe <- function(fixture) {
      a <- cudaverse::cuda_tensor(fixture$left, device = "cuda", dtype = "float32")
      b <- cudaverse::cuda_tensor(fixture$right, device = "cuda", dtype = "float32")
      before <- cudaverse::tensor_matmul(a, b, precision = "standard")
      tf32 <- cudaverse::tensor_matmul(a, b, precision = "tf32")
      after <- cudaverse::tensor_matmul(a, b, precision = "standard")
      after_default <- cudaverse::tensor_matmul(a, b)
      a64 <- cudaverse::cuda_tensor(fixture$left, device = "cuda", dtype = "float64")
      b64 <- cudaverse::cuda_tensor(fixture$right, device = "cuda", dtype = "float64")
      double <- cudaverse::tensor_matmul(a64, b64, precision = "standard")
      factory$synchronize()
      before_host <- cudaverse::to_cpu(before)
      standard_hash <- value_sha(before_host)
      checks <- list(
        standard_after_tf32_bitwise_unchanged = identical(standard_hash, value_sha(cudaverse::to_cpu(after))),
        default_after_tf32_bitwise_unchanged = identical(standard_hash, value_sha(cudaverse::to_cpu(after_default))),
        original_input_casts_exact = identical(cudaverse::to_cpu(a), fixture$left) &&
          identical(cudaverse::to_cpu(b), fixture$right) &&
          identical(cudaverse::to_cpu(a64), fixture$left) && identical(cudaverse::to_cpu(b64), fixture$right)
      )
      results <- list()
      tensors <- list(standard_float32 = before, tf32_permitted_float32 = tf32, standard_float64 = double)
      for (name in names(modes)) {
        host <- cudaverse::to_cpu(tensors[[name]])
        metadata <- inspect_tensor(tensors[[name]], modes[[name]], fixture)
        validation <- validate_mode(host, modes[[name]], fixture)
        exact <- !fixture$exact || identical(host, fixture$reference)
        results[[name]] <- list(
          passed = isTRUE(validation$passed) && metadata$passed && exact,
          validation = validation, metadata = metadata, host_value_sha256 = value_sha(host),
          exact_control_required = fixture$exact, exact_control_passed = exact)
      }
      list(passed = all(unlist(checks)) && all(vapply(results, function(x) isTRUE(x$passed), logical(1))),
           checks = checks, modes = results,
           tracked_memory_with_results = cudaverse:::.native_memory_tracker())
    }
    summarize_times <- function(values) {
      require_check(all(is.finite(values)) && all(values >= 0),
                    "The wall clock produced an invalid sample duration.")
      list(median_seconds = unname(stats::median(values)),
           p95_seconds = unname(stats::quantile(values, 0.95, type = 8)),
           runs_seconds = as.numeric(values))
    }
    time_mode <- function(fixture, name, expected_hash) {
      mode <- modes[[name]]
      x <- cudaverse::cuda_tensor(fixture$left, device = "cuda", dtype = mode$dtype)
      y <- cudaverse::cuda_tensor(fixture$right, device = "cuda", dtype = mode$dtype)
      factory$synchronize()
      resident_run <- function() {
        factory$synchronize()
        value <- cudaverse::tensor_matmul(x, y, precision = mode$precision)
        factory$synchronize()
        value
      }
      host_run <- function() {
        factory$synchronize()
        left <- cudaverse::cuda_tensor(fixture$left, device = "cuda", dtype = mode$dtype)
        right <- cudaverse::cuda_tensor(fixture$right, device = "cuda", dtype = mode$dtype)
        value <- cudaverse::tensor_matmul(left, right, precision = mode$precision)
        factory$synchronize()
        host <- cudaverse::to_cpu(value)
        factory$synchronize()
        list(tensor = value, host = host)
      }
      measure_scope <- function(scope, run) {
        samples <- list()
        guards <- list()
        guard <- if (full) function(context) {
          pids <- benchmark_gpu_compute_pids()
          guards[[length(guards) + 1L]] <<- list(context = context, compute_pids = pids)
          benchmark_assert_idle_gpu(paste(fixture$id, name, scope, context, sep = "/"),
                                    pids = pids, current_pid = Sys.getpid())
        } else NULL
        observe <- function(value) {
          tensor <- if (scope == "host_boundary") value$tensor else value
          host <- if (scope == "host_boundary") value$host else cudaverse::to_cpu(value)
          metadata <- inspect_tensor(tensor, mode, fixture)
          hash <- value_sha(host)
          list(passed = metadata$passed && identical(hash, expected_hash),
               matches_validated_probe_bitwise = identical(hash, expected_hash),
               host_value_sha256 = hash, metadata = metadata)
        }
        timing <- tryCatch(benchmark_time_runs(
          cold_run = run, timed_run = run, warmups = warmups, timed_runs = timed_runs,
          summarize = summarize_times, collect = observe,
          clock = function() as.numeric(Sys.time()), guard = guard,
          progress = function(event, index, total, seconds) {
            samples[[length(samples) + 1L]] <<- list(event = event, index = index,
                                                    total = total, seconds = seconds)
          }
        ), error = function(e) list(error = conditionMessage(e)),
        interrupt = function(e) list(error = "interrupted"))
        timing$last <- NULL
        complete <- is.null(timing$error) && length(timing$observations) == timed_runs
        passed <- complete && all(vapply(timing$observations, function(x) isTRUE(x$passed), logical(1)))
        guard_count <- if (full) 2L * (1L + warmups + timed_runs) else 0L
        list(passed = passed && length(guards) == guard_count,
             result = timing, sample_events = samples, idle_guard_observations = guards,
             expected_idle_guard_observations = guard_count)
      }
      # Validation downloads, sample hashes, and GC are outside the timer.
      host <- measure_scope("host_boundary", host_run)
      if (!isTRUE(host$passed)) return(list(passed = FALSE, host_boundary = host))
      resident <- measure_scope("resident_compute", resident_run)
      list(passed = isTRUE(host$passed) && isTRUE(resident$passed),
           host_boundary = host, resident_compute = resident,
           tracked_memory_with_resident_inputs = cudaverse:::.native_memory_tracker())
    }
    for (id in case_ids) {
      cat(sprintf("PRECISION_CASE=%s\n", id))
      inputs <- make_fixture(id)
      entry <- list(id = id, seed = inputs$seed, exact_control = inputs$exact,
                    shapes = list(left = dim(inputs$left), right = dim(inputs$right)),
                    input_sha256 = object_sha(list(left = inputs$left, right = inputs$right)),
                    reference_value_sha256 = value_sha(inputs$reference), timing = list())
      if (identical(id, "matmul-float32-64")) {
        require_check(identical(entry$input_sha256,
          "52bf0872d9647cb8972a370505694065dda86da425c07bf8c26c8f1eeefa5571"),
          "Original 64x64 benchmark inputs changed.")
      }
      report$cases[[id]] <- entry
      if (full) benchmark_assert_idle_gpu(paste(id, "before correctness probe"))
      entry$accuracy <- with_memory_check(function() correctness_probe(inputs))
      report$cases[[id]] <- entry
      if (full) benchmark_assert_idle_gpu(paste(id, "after correctness probe"))
      if (!isTRUE(entry$accuracy$passed)) break
      for (name in names(modes)) {
        cat(sprintf("  PRECISION_MODE=%s\n", name))
        measured <- with_memory_check(function() time_mode(
          inputs, name, entry$accuracy$modes[[name]]$host_value_sha256))
        entry$timing[[name]] <- measured
        report$cases[[id]] <- entry
        if (!isTRUE(measured$passed)) break
      }
      entry$passed <- isTRUE(entry$accuracy$passed) && length(entry$timing) == length(modes) &&
        all(vapply(entry$timing, function(x) isTRUE(x$passed), logical(1)))
      report$cases[[id]] <- entry
      if (!entry$passed) break
    }
    factory$synchronize()
    invisible(gc())
    report$final_memory <- cudaverse:::.native_memory_tracker()
    report$global_cleanup_passed <- identical(as.numeric(report$initial_memory$current),
                                              as.numeric(report$final_memory$current))
    report$source$status_after <- command_result("git", c("status", "--porcelain", "--untracked-files=all"))
    report$source$status_unchanged <- identical(report$source$status, report$source$status_after)
    report$source$commit_after <- command_result("git", c("rev-parse", "HEAD"))
    report$source$commit_unchanged <- identical(report$source$commit, report$source$commit_after)
    report$source$file_sha256_after <- as.list(setNames(vapply(source_files, sha, character(1)), source_files))
    report$source$files_unchanged <- identical(report$source$file_sha256, report$source$file_sha256_after)
    report$installed$dll_sha256_after <- sha(dll_path)
    report$installed$dll_unchanged <- identical(report$installed$dll_sha256, report$installed$dll_sha256_after)
    report$installed$source_identity_after <- if (exists(
      "benchmark_installed_source_identity", envir = environment(), inherits = FALSE
    )) tryCatch(
      benchmark_installed_source_identity(source_root = ".", package = "cudaverse"),
      error = function(e) list(verified = FALSE, error = conditionMessage(e))) else
        list(verified = FALSE, reason = "source identity helper is unavailable")
    identity_fields <- c("verified", "reason", "source_commit", "source_tree",
                         "manifest_sha256", "installed_payload_sha256", "installed_package")
    report$installed$source_identity_unchanged <- all(vapply(
      identity_fields, function(field) identical(
        report$installed$source_identity[[field]],
        report$installed$source_identity_after[[field]]), logical(1)))
    report$complete <- identical(names(report$cases), case_ids)
    report$passed <- report$complete && report$global_cleanup_passed &&
      all(vapply(report$cases, function(x) isTRUE(x$passed), logical(1)))
    if (full) report$passed <- report$passed && isTRUE(report$source$status_unchanged) &&
      isTRUE(report$source$commit_unchanged) && isTRUE(report$source$files_unchanged) &&
      isTRUE(report$installed$dll_unchanged) &&
      isTRUE(report$installed$source_identity_after$verified) &&
      isTRUE(report$installed$source_identity_unchanged)
    report$retained_performance_eligible <- full && report$passed && report$source$clean
    report$environment$compute_processes_at_finish <- command_result("nvidia-smi", c(
      "--query-compute-apps=pid,process_name", "--format=csv,noheader"))
  }, error = fail_report, interrupt = fail_report)
  report$session <- capture.output(sessionInfo())
  report$finished_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
  if (file.exists(output)) stop("Refusing to overwrite a report created during this run.", call. = FALSE)
  jsonlite::write_json(report, output, pretty = TRUE, auto_unbox = TRUE,
                       digits = 17, force = TRUE, null = "null", na = "null")
  cat(sprintf("PRECISION_REPORT=%s\n", output))
  if (!isTRUE(report$passed)) {
    stop("The precision comparison failed; diagnostics were preserved in the JSON report.", call. = FALSE)
  }
  cat(sprintf("PRECISION_EXPERIMENT=PASSED profile=%s retained_performance_eligible=%s\n",
              profile, report$retained_performance_eligible))
  invisible(report)
}

run_matmul_precision_experiment()
