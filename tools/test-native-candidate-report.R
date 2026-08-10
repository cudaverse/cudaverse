sys.source(
  file.path("tools", "native-candidate-report-io.R"),
  envir = environment()
)
if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Native candidate report self-tests require jsonlite and digest.",
       call. = FALSE)
}

expect_failure <- function(code, pattern) {
  error <- tryCatch({
    force(code)
    NULL
  }, error = identity)
  if (!inherits(error, "error") ||
      !grepl(pattern, conditionMessage(error), fixed = TRUE)) {
    stop("Expected an error containing: ", pattern, call. = FALSE)
  }
}
commit <- paste(rep("a", 40L), collapse = "")
version <- "0.3.0.9000"
passed <- function(...) c(list(passed = TRUE), list(...))

consolidation <- list(
  schema = "cudaverse-native-consolidation/1",
  source = list(cudaverse = list(
    commit = commit, tracked_dirty = FALSE
  )),
  hardware = list(nvidia_smi = "NVIDIA RTX 2000 Ada Generation Laptop GPU"),
  software = list(cudaverse = version),
  benchmarks = list(case = list(
    base = list(status = "complete", validation = passed()),
    native = list(status = "complete", validation = passed()),
    torch = list(status = "complete", validation = passed())
  )),
  hardware_validation = list(
    auto_selection = list(contract_schema = "cudaverse-backend/1"),
    dtype_surface = passed(),
    resident_pipeline = passed(
      provenance_schema = c(
        normalized = "cudaverse-stage/1",
        pca = "cudaverse-stage/1",
        knn = "cudaverse-stage/1"
      )
    ),
    stable_ties = passed(),
    shared_ownership = passed(),
    structured_error = passed(backend_reusable = TRUE),
    injected_cuda_error = passed(),
    interruption = passed(backend_reusable = TRUE),
    dense_lifecycle = passed(
      cycles = 1000, whole_device_absolute_difference_bytes = 0,
      tracked_current_difference_bytes = 0
    ),
    lifecycle = passed(
      cycles = 1000, whole_device_absolute_difference_bytes = 0,
      tracked_current_difference_bytes = 0
    )
  ),
  overall_pass = TRUE
)
package_tests <- list(
  schema = "cudaverse-native-package-tests/1",
  source = list(cudaverse = list(
    commit = commit, tracked_dirty = FALSE
  )),
  hardware = list(nvidia_smi = "NVIDIA RTX 2000 Ada Generation Laptop GPU"),
  software = list(
    R = R.version.string, cudaverse = version,
    native_diagnostics = list(
      available = TRUE, runtime_complete = TRUE,
      self_test = list(passed = TRUE)
    )
  ),
  testthat = list(
    tests = 10, expectations = 100, failures = 0, errors = 0, skips = 0,
    native_ready = TRUE, passed = TRUE
  ),
  overall_pass = TRUE
)

build <- function(x = consolidation, y = package_tests) {
  build_native_candidate_report(x, y, paste(rep("b", 64L), collapse = ""),
                                paste(rep("c", 64L), collapse = ""))
}
report <- build()
stopifnot(!length(validate_native_candidate_report(
  report, expected_commit = commit, expected_version = version
)))

changed <- package_tests
changed$source$cudaverse$commit <- paste(rep("d", 40L), collapse = "")
expect_failure(build(y = changed), "different commits")
changed <- package_tests
changed$software$cudaverse <- "0.3.0"
expect_failure(build(y = changed), "different versions")

reject_report <- function(mutator, pattern) {
  changed <- report
  changed <- mutator(changed)
  failures <- validate_native_candidate_report(changed)
  if (!any(grepl(pattern, failures, fixed = TRUE))) {
    stop("Expected validation failure containing: ", pattern, call. = FALSE)
  }
}
reject_report(function(x) {
  x$gates$no_skips <- FALSE; x
}, "no_skips")
reject_report(function(x) {
  x$lifecycle$dense$post_cleanup_difference_bytes <- 1024^2 + 1; x
}, "exceeds the 1 MiB ceiling")
reject_report(function(x) {
  x$contracts$backend <- "wrong"; x
}, "backend contract is invalid")
reject_report(function(x) {
  x$contracts$stage <- "wrong"; x
}, "stage contract is invalid")
reject_report(function(x) {
  x$overall_pass <- FALSE; x
}, "overall gate failed")

work <- tempfile("cudaverse-native-candidate-")
dir.create(work)
consolidation_path <- file.path(work, "consolidation.json")
package_tests_path <- file.path(work, "package-tests.json")
candidate_path <- file.path(work, "candidate.json")
jsonlite::write_json(
  consolidation, consolidation_path, auto_unbox = TRUE, pretty = TRUE
)
jsonlite::write_json(
  package_tests, package_tests_path, auto_unbox = TRUE, pretty = TRUE
)
variables <- c(
  CUDAVERSE_CONSOLIDATION_REPORT = consolidation_path,
  CUDAVERSE_NATIVE_PACKAGE_TEST_REPORT = package_tests_path,
  CUDAVERSE_NATIVE_CANDIDATE_REPORT = candidate_path
)
old <- Sys.getenv(names(variables), unset = NA_character_)
on.exit({
  Sys.unsetenv(names(variables))
  restore <- !is.na(old)
  if (any(restore)) do.call(Sys.setenv, as.list(old[restore]))
}, add = TRUE)
do.call(Sys.setenv, as.list(variables))
sys.source(
  file.path("tools", "build-native-candidate-report.R"),
  envir = new.env(parent = globalenv())
)
stopifnot(file.exists(candidate_path))
sys.source(
  file.path("tools", "check-native-candidate-report.R"),
  envir = new.env(parent = globalenv())
)

message("Native candidate report positive and rejection self-tests passed.")
