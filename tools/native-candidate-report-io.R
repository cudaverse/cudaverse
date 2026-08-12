sys.source(
  file.path("tools", "candidate-policy.R"),
  envir = environment()
)

native_candidate_scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value) || is.na(value[[1L]])) default else value[[1L]]
}

native_candidate_logical <- function(x) {
  isTRUE(as.logical(native_candidate_scalar(x, FALSE)))
}

native_candidate_number <- function(x) {
  suppressWarnings(as.numeric(native_candidate_scalar(x, NA_real_)))
}

native_candidate_source <- function(report) report$source$cudaverse

native_candidate_text_vector <- function(x) {
  as.character(unlist(x, recursive = TRUE, use.names = FALSE))
}

native_candidate_all_benchmarks_pass <- function(report) {
  length(report$benchmarks) > 0L && all(vapply(
    report$benchmarks,
    function(case) length(case) > 0L && all(vapply(
      case,
      function(value) {
        identical(native_candidate_scalar(value$status, ""), "complete") &&
          native_candidate_logical(value$validation$passed)
      },
      logical(1L)
    )),
    logical(1L)
  ))
}

build_native_candidate_report <- function(consolidation, package_tests,
                                          consolidation_sha256,
                                          package_tests_sha256) {
  if (!identical(
    native_candidate_scalar(consolidation$schema, ""),
    "cudaverse-native-consolidation/1"
  )) stop("The RTX input is not a consolidation report.", call. = FALSE)
  if (!identical(
    native_candidate_scalar(package_tests$schema, ""),
    "cudaverse-native-package-tests/1"
  )) stop("The package-test input has an unexpected schema.", call. = FALSE)

  source <- native_candidate_source(consolidation)
  test_source <- native_candidate_source(package_tests)
  commit <- as.character(native_candidate_scalar(source$commit, ""))
  test_commit <- as.character(native_candidate_scalar(test_source$commit, ""))
  version <- as.character(native_candidate_scalar(
    consolidation$software$cudaverse, ""
  ))
  test_version <- as.character(native_candidate_scalar(
    package_tests$software$cudaverse, ""
  ))
  if (!identical(commit, test_commit)) {
    stop("RTX consolidation and package tests use different commits.",
         call. = FALSE)
  }
  if (!identical(version, test_version)) {
    stop("RTX consolidation and package tests use different versions.",
         call. = FALSE)
  }
  hardware <- native_candidate_text_vector(
    consolidation$hardware$nvidia_smi
  )
  test_hardware <- native_candidate_text_vector(
    package_tests$hardware$nvidia_smi
  )
  if (!length(hardware) || !identical(hardware, test_hardware) ||
      !grepl("RTX 2000", paste(hardware, collapse = " "), fixed = TRUE)) {
    stop(
      "RTX consolidation and package tests must identify the same RTX 2000.",
      call. = FALSE
    )
  }
  if (!identical(
    as.character(native_candidate_scalar(consolidation$software$R, "")),
    as.character(native_candidate_scalar(package_tests$software$R, ""))
  )) {
    stop("RTX consolidation and package tests use different R runtimes.",
         call. = FALSE)
  }
  if (native_candidate_logical(source$tracked_dirty) ||
      native_candidate_logical(test_source$tracked_dirty)) {
    stop("Native candidate inputs must come from clean source.", call. = FALSE)
  }

  validation <- consolidation$hardware_validation
  dense <- validation$dense_lifecycle
  sparse <- validation$lifecycle
  benchmark_parity <- native_candidate_all_benchmarks_pass(consolidation)
  parity <- native_candidate_logical(consolidation$overall_pass) &&
    benchmark_parity &&
    native_candidate_logical(validation$dtype_surface$passed) &&
    native_candidate_logical(validation$resident_pipeline$passed) &&
    native_candidate_logical(validation$stable_ties$passed)
  structured_recovery <-
    native_candidate_logical(validation$shared_ownership$passed) &&
    native_candidate_logical(validation$structured_error$passed) &&
    native_candidate_logical(validation$injected_cuda_error$passed)
  interruption <- native_candidate_logical(validation$interruption$passed)
  backend_reuse <-
    native_candidate_logical(validation$structured_error$backend_reusable) &&
    native_candidate_logical(validation$interruption$backend_reusable)
  no_skips <- native_candidate_logical(package_tests$testthat$passed) &&
    native_candidate_logical(package_tests$testthat$native_ready) &&
    native_candidate_logical(package_tests$overall_pass) &&
    native_candidate_number(package_tests$testthat$tests) > 0 &&
    native_candidate_number(package_tests$testthat$expectations) > 0 &&
    identical(native_candidate_number(package_tests$testthat$failures), 0) &&
    identical(native_candidate_number(package_tests$testthat$errors), 0) &&
    identical(native_candidate_number(package_tests$testthat$skips), 0)

  lifecycle <- list(
    dense = list(
      cycles = native_candidate_number(dense$cycles),
      post_cleanup_difference_bytes = native_candidate_number(
        dense$whole_device_absolute_difference_bytes
      ),
      tracked_current_difference_bytes = native_candidate_number(
        dense$tracked_current_difference_bytes
      ),
      no_double_free = native_candidate_logical(
        validation$shared_ownership$passed
      ),
      passed = native_candidate_logical(dense$passed)
    ),
    sparse = list(
      cycles = native_candidate_number(sparse$cycles),
      post_cleanup_difference_bytes = native_candidate_number(
        sparse$whole_device_absolute_difference_bytes
      ),
      tracked_current_difference_bytes = native_candidate_number(
        sparse$tracked_current_difference_bytes
      ),
      no_double_free = native_candidate_logical(
        validation$shared_ownership$passed
      ),
      passed = native_candidate_logical(sparse$passed)
    )
  )
  gates <- list(
    parity = parity,
    structured_recovery = structured_recovery,
    interruption = interruption,
    backend_reuse = backend_reuse,
    no_skips = no_skips
  )

  report <- list(
    schema = "cudaverse-native-candidate/1",
    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    source = list(cudaverse = list(
      commit = commit, tracked_dirty = FALSE
    )),
    hardware = list(nvidia_smi = test_hardware),
    software = list(
      R = package_tests$software$R,
      cudaverse = version,
      native_diagnostics = package_tests$software$native_diagnostics
    ),
    contracts = list(
      backend = validation$auto_selection$contract_schema,
      stage = unique(unlist(
        validation$resident_pipeline$provenance_schema,
        recursive = TRUE, use.names = FALSE
      ))
    ),
    inputs = list(
      consolidation = list(
        schema = consolidation$schema, sha256 = consolidation_sha256
      ),
      package_tests = list(
        schema = package_tests$schema, sha256 = package_tests_sha256
      )
    ),
    gates = gates,
    lifecycle = lifecycle,
    overall_pass = all(vapply(gates, isTRUE, logical(1L))) &&
      all(vapply(lifecycle, function(value) {
        isTRUE(value$passed) && value$cycles >= 1000 &&
          value$post_cleanup_difference_bytes <= 1024^2 &&
          identical(value$tracked_current_difference_bytes, 0) &&
          isTRUE(value$no_double_free)
      }, logical(1L)))
  )
  report
}

validate_native_candidate_report <- function(report, expected_commit = NULL,
                                             expected_version = NULL) {
  failures <- character()
  require_gate <- function(value, message) {
    if (!isTRUE(value)) failures <<- c(failures, message)
  }
  commit <- as.character(native_candidate_scalar(
    report$source$cudaverse$commit, ""
  ))
  version <- as.character(native_candidate_scalar(
    report$software$cudaverse, ""
  ))
  require_gate(
    identical(native_candidate_scalar(report$schema, ""),
              "cudaverse-native-candidate/1"),
    "unexpected native candidate schema"
  )
  require_gate(grepl("^[0-9a-f]{40}$", commit),
               "native candidate commit is invalid")
  require_gate(!native_candidate_logical(
    report$source$cudaverse$tracked_dirty
  ), "native candidate source is dirty")
  supported_version <- any(vapply(
    candidate_release_policies(),
    function(policy) grepl(policy$version_pattern, version),
    logical(1L)
  ))
  require_gate(supported_version, "native candidate version is invalid")
  if (!is.null(expected_commit)) {
    require_gate(identical(commit, expected_commit),
                 "native candidate commit does not match")
  }
  if (!is.null(expected_version)) {
    require_gate(identical(version, expected_version),
                 "native candidate version does not match")
  }
  require_gate(
    grepl("RTX 2000", paste(unlist(
      report$hardware$nvidia_smi, recursive = TRUE, use.names = FALSE
    ), collapse = " "), fixed = TRUE),
    "native candidate was not run on the RTX 2000"
  )
  require_gate(
    native_candidate_logical(report$software$native_diagnostics$available) &&
      native_candidate_logical(
        report$software$native_diagnostics$runtime_complete
      ) &&
      native_candidate_logical(
        report$software$native_diagnostics$self_test$passed
      ),
    "native candidate diagnostics do not prove a usable native runtime"
  )
  require_gate(
    identical(native_candidate_scalar(report$contracts$backend, ""),
              "cudaverse-backend/1"),
    "native candidate backend contract is invalid"
  )
  require_gate(
    identical(unlist(
      report$contracts$stage, recursive = TRUE, use.names = FALSE
    ), "cudaverse-stage/1"),
    "native candidate stage contract is invalid"
  )
  input_contract <- list(
    consolidation = "cudaverse-native-consolidation/1",
    package_tests = "cudaverse-native-package-tests/1"
  )
  for (name in names(input_contract)) {
    require_gate(
      identical(native_candidate_scalar(
        report$inputs[[name]]$schema, ""
      ), input_contract[[name]]),
      paste("native candidate input schema is invalid:", name)
    )
    require_gate(
      grepl("^[0-9a-f]{64}$", as.character(native_candidate_scalar(
        report$inputs[[name]]$sha256, ""
      ))),
      paste("native candidate input SHA-256 is invalid:", name)
    )
  }
  for (gate in c("parity", "structured_recovery", "interruption",
                 "backend_reuse", "no_skips")) {
    require_gate(native_candidate_logical(report$gates[[gate]]),
                 paste("native candidate gate failed:", gate))
  }
  for (kind in c("dense", "sparse")) {
    value <- report$lifecycle[[kind]]
    require_gate(native_candidate_number(value$cycles) >= 1000,
                 paste(kind, "candidate lifecycle has fewer than 1,000 cycles"))
    require_gate(
      native_candidate_number(value$post_cleanup_difference_bytes) <= 1024^2,
      paste(kind, "candidate lifecycle exceeds the 1 MiB ceiling")
    )
    require_gate(
      identical(native_candidate_number(
        value$tracked_current_difference_bytes
      ), 0),
      paste(kind, "candidate lifecycle retains tracked bytes")
    )
    require_gate(native_candidate_logical(value$no_double_free) &&
                   native_candidate_logical(value$passed),
                 paste(kind, "candidate lifecycle did not pass safely"))
  }
  require_gate(native_candidate_logical(report$overall_pass),
               "native candidate overall gate failed")
  unique(failures)
}
