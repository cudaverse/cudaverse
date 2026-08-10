if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking candidate evidence requires jsonlite.", call. = FALSE)
}

input <- Sys.getenv("CUDAVERSE_CANDIDATE_MANIFEST", unset = "")
if (!nzchar(input) || !file.exists(input)) {
  stop("Set CUDAVERSE_CANDIDATE_MANIFEST to an existing manifest.",
       call. = FALSE)
}

manifest <- jsonlite::read_json(input, simplifyVector = FALSE)
scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value)) default else value[[1L]]
}
text_value <- function(x) as.character(scalar(x, ""))
number <- function(x) suppressWarnings(as.numeric(scalar(x, NA_real_)))
logical_value <- function(x) isTRUE(as.logical(scalar(x, FALSE)))
is_sha256 <- function(x) grepl("^[0-9a-f]{64}$", text_value(x))
is_commit <- function(x) grepl("^[0-9a-f]{40}$", text_value(x))

failures <- character()
require_gate <- function(value, message) {
  if (!isTRUE(value)) failures <<- c(failures, message)
}

require_gate(
  identical(text_value(manifest$schema), "cudaverse-candidate-evidence/1"),
  "unexpected candidate-evidence schema"
)
commit <- text_value(manifest$candidate$source_commit)
require_gate(is_commit(commit), "candidate source commit is invalid")
version <- text_value(manifest$candidate$version)
require_gate(
  grepl("^0\\.3\\.0(\\.9000)?$", version),
  "candidate version is not a 0.3.0 candidate"
)
require_gate(
  identical(text_value(manifest$candidate$branch), "develop/native-cuda"),
  "candidate branch is not develop/native-cuda"
)
require_gate(logical_value(manifest$candidate$clean),
             "candidate source tree is not recorded as clean")
require_gate(
  identical(text_value(manifest$frozen_refs$main),
            "59e15c8c5a56d26e09a594886c875b1b8249f6f9") &&
    identical(text_value(manifest$frozen_refs$release_cran_0_1_0),
              "59e15c8c5a56d26e09a594886c875b1b8249f6f9"),
  "frozen release refs changed"
)

require_gate(
  identical(text_value(manifest$source_tarball$source_commit), commit),
  "source tarball does not identify the candidate commit"
)
require_gate(
  identical(text_value(manifest$source_tarball$file),
            paste0("cudaverse_", version, ".tar.gz")),
  "source tarball name does not match the candidate version"
)
require_gate(is_sha256(manifest$source_tarball$sha256),
             "source tarball SHA-256 is invalid")
require_gate(is_sha256(manifest$source_tarball$check_log_sha256),
             "local check-log SHA-256 is invalid")
require_gate(
  identical(number(manifest$source_tarball$check$errors), 0) &&
    identical(number(manifest$source_tarball$check$warnings), 0),
  "local source check has errors or warnings"
)

required_checks <- c(
  "windows_r_release", "macos_r_release", "ubuntu_r_release",
  "ubuntu_r_devel", "pkgdown", "cpu_contract", "supply_chain",
  "cuda_12_8_1_abi", "cuda_12_8_1_ptx", "windows_artifact",
  "linux_artifact"
)
require_gate(
  length(manifest$github_checks) == length(required_checks) &&
    identical(sort(names(manifest$github_checks)), sort(required_checks)),
  "GitHub check inventory does not match the candidate contract"
)
for (name in required_checks) {
  check <- manifest$github_checks[[name]]
  require_gate(!is.null(check), paste(name, "check is missing"))
  if (is.null(check)) next
  require_gate(
    identical(text_value(check$source_commit), commit),
    paste(name, "check is from another source commit")
  )
  require_gate(
    identical(text_value(check$conclusion), "success"),
    paste(name, "check did not succeed")
  )
  require_gate(nzchar(text_value(check$url)), paste(name, "check URL is missing"))
}

for (name in c("windows", "linux")) {
  artifact <- manifest$artifacts[[name]]
  require_gate(!is.null(artifact), paste(name, "artifact is missing"))
  if (is.null(artifact)) next
  require_gate(
    identical(text_value(artifact$source_commit), commit),
    paste(name, "artifact is from another source commit")
  )
  require_gate(is_sha256(artifact$sha256),
               paste(name, "artifact SHA-256 is invalid"))
  require_gate(number(artifact$bytes) > 0,
               paste(name, "artifact size is invalid"))
}

require_gate(
  identical(text_value(manifest$supply_chain$source_commit), commit),
  "supply-chain evidence is from another source commit"
)
require_gate(is_sha256(manifest$supply_chain$sbom_sha256),
             "SBOM SHA-256 is invalid")
require_gate(is_sha256(manifest$supply_chain$license_inventory_sha256),
             "license-inventory SHA-256 is invalid")
require_gate(
  identical(number(manifest$supply_chain$bundled_nvidia_runtime_bytes), 0) &&
    identical(number(manifest$supply_chain$bundled_libtorch_bytes), 0),
  "candidate bundles an NVIDIA runtime or LibTorch"
)

require_gate(
  identical(text_value(manifest$rtx$source_commit), commit),
  "RTX evidence is from another source commit"
)
require_gate(is_sha256(manifest$rtx$report_sha256),
             "RTX report SHA-256 is invalid")
for (gate in c("parity", "structured_recovery", "interruption",
               "backend_reuse", "no_skips")) {
  require_gate(logical_value(manifest$rtx[[gate]]),
               paste("RTX", gate, "gate did not pass"))
}
for (kind in c("dense", "sparse")) {
  lifecycle <- manifest$rtx$lifecycle[[kind]]
  require_gate(number(lifecycle$cycles) >= 1000,
               paste(kind, "lifecycle has fewer than 1,000 cycles"))
  require_gate(number(lifecycle$post_cleanup_difference_bytes) <= 1024^2,
               paste(kind, "lifecycle exceeds the 1 MiB cleanup ceiling"))
  require_gate(logical_value(lifecycle$no_double_free),
               paste(kind, "lifecycle does not prove no double-free"))
}

require_gate(
  identical(text_value(manifest$benchmark$source_commit), commit),
  "benchmark evidence is from another source commit"
)
require_gate(is_sha256(manifest$benchmark$report_sha256),
             "benchmark report SHA-256 is invalid")
require_gate(is_sha256(manifest$benchmark$summary_sha256),
             "benchmark summary SHA-256 is invalid")
require_gate(logical_value(manifest$benchmark$complete) &&
               logical_value(manifest$benchmark$report_checker_passed) &&
               logical_value(manifest$benchmark$summary_checker_passed),
             "benchmark report or summary is incomplete or unchecked")

require_gate(
  identical(text_value(manifest$documentation$source_commit), commit),
  "documentation evidence is from another source commit"
)
require_gate(logical_value(manifest$documentation$pkgdown_passed) &&
               logical_value(manifest$documentation$render_reviewed),
             "pkgdown output was not both built and reviewed")
require_gate(!logical_value(manifest$documentation$pages_deployed),
             "candidate evidence records an unauthorized Pages deployment")

outcome <- text_value(manifest$decision$outcome)
require_gate(outcome %in% c("release", "defer", "reduce-scope"),
             "candidate decision is missing or invalid")
require_gate(
  identical(text_value(manifest$decision$source_commit), commit),
  "candidate decision is from another source commit"
)
require_gate(length(manifest$decision$limitations) > 0,
             "candidate decision does not record remaining limitations")
require_gate(!logical_value(manifest$decision$external_release_action_taken),
             "an external release action was taken before approval")

if (length(failures)) {
  stop(
    "Candidate evidence failed:\n- ",
    paste(unique(failures), collapse = "\n- "),
    call. = FALSE
  )
}
message("Candidate evidence matches one exact review source: ", commit)
