if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Candidate-evidence self-tests require jsonlite.", call. = FALSE)
}

run_checker <- function() {
  sys.source(
    file.path("tools", "check-candidate-evidence.R"),
    envir = new.env(parent = globalenv())
  )
}
expect_error_message <- function(code, pattern) {
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
sha <- paste(rep("b", 64L), collapse = "")
check <- function() list(
  source_commit = commit,
  conclusion = "success",
  url = "https://example.invalid/check"
)
artifact <- function() list(source_commit = commit, sha256 = sha, bytes = 1L)
lifecycle <- function() list(
  cycles = 1000L,
  post_cleanup_difference_bytes = 0L,
  no_double_free = TRUE
)

manifest <- list(
  schema = "cudaverse-candidate-evidence/1",
  candidate = list(
    source_commit = commit, version = "0.3.0.9000",
    branch = "develop/native-cuda", clean = TRUE
  ),
  frozen_refs = list(
    main = "59e15c8c5a56d26e09a594886c875b1b8249f6f9",
    release_cran_0_1_0 = "59e15c8c5a56d26e09a594886c875b1b8249f6f9"
  ),
  source_tarball = list(
    source_commit = commit, file = "cudaverse_0.3.0.9000.tar.gz",
    sha256 = sha, check_log_sha256 = sha,
    check = list(errors = 0L, warnings = 0L)
  ),
  github_checks = setNames(
    replicate(11L, check(), simplify = FALSE),
    c(
      "windows_r_release", "macos_r_release", "ubuntu_r_release",
      "ubuntu_r_devel", "pkgdown", "cpu_contract", "supply_chain",
      "cuda_12_8_1_abi", "cuda_12_8_1_ptx", "windows_artifact",
      "linux_artifact"
    )
  ),
  artifacts = list(windows = artifact(), linux = artifact()),
  supply_chain = list(
    source_commit = commit, sbom_sha256 = sha,
    license_inventory_sha256 = sha,
    bundled_nvidia_runtime_bytes = 0L, bundled_libtorch_bytes = 0L
  ),
  rtx = list(
    source_commit = commit, report_sha256 = sha, parity = TRUE,
    structured_recovery = TRUE, interruption = TRUE, backend_reuse = TRUE,
    no_skips = TRUE,
    lifecycle = list(dense = lifecycle(), sparse = lifecycle())
  ),
  benchmark = list(
    source_commit = commit, report_sha256 = sha, summary_sha256 = sha,
    complete = TRUE, report_checker_passed = TRUE,
    summary_checker_passed = TRUE
  ),
  documentation = list(
    source_commit = commit, pkgdown_passed = TRUE, render_reviewed = TRUE,
    pages_deployed = FALSE
  ),
  decision = list(
    source_commit = commit, outcome = "defer",
    limitations = list("synthetic limitation"),
    external_release_action_taken = FALSE
  )
)

work <- tempfile("cudaverse-candidate-evidence-")
dir.create(work)
path <- file.path(work, "manifest.json")
write_manifest <- function() {
  jsonlite::write_json(
    manifest, path, auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
}
Sys.setenv(CUDAVERSE_CANDIDATE_MANIFEST = path)

write_manifest()
run_checker()

manifest$benchmark$source_commit <- paste(rep("c", 40L), collapse = "")
write_manifest()
expect_error_message(run_checker(), "benchmark evidence is from another")
manifest$benchmark$source_commit <- commit

manifest$github_checks$ubuntu_r_devel$conclusion <- "skipped"
write_manifest()
expect_error_message(run_checker(), "ubuntu_r_devel check did not succeed")
manifest$github_checks$ubuntu_r_devel$conclusion <- "success"

manifest$rtx$lifecycle$sparse$post_cleanup_difference_bytes <- 1024^2 + 1L
write_manifest()
expect_error_message(run_checker(), "sparse lifecycle exceeds")
manifest$rtx$lifecycle$sparse$post_cleanup_difference_bytes <- 0L

manifest$supply_chain$bundled_libtorch_bytes <- 1L
write_manifest()
expect_error_message(run_checker(), "bundles an NVIDIA runtime or LibTorch")
manifest$supply_chain$bundled_libtorch_bytes <- 0L

manifest$documentation$pages_deployed <- TRUE
write_manifest()
expect_error_message(run_checker(), "unauthorized Pages deployment")

message("Candidate-evidence positive and rejection self-tests passed.")
