if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Checking candidate evidence requires jsonlite and digest.",
       call. = FALSE)
}
sys.source(
  file.path("tools", "native-candidate-report-io.R"),
  envir = environment()
)
sys.source(
  file.path("tools", "native-session-report-io.R"),
  envir = environment()
)
sys.source(
  file.path("tools", "candidate-policy.R"),
  envir = environment()
)
sys.source(
  file.path("tools", "check-redistributables.R"),
  envir = environment()
)

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
bundle_root <- normalizePath(
  dirname(input), winslash = "/", mustWork = TRUE
)
check_evidence_file <- function(path_value, sha_value, label) {
  relative <- text_value(path_value)
  require_gate(nzchar(relative), paste(label, "path is missing"))
  path_components <- strsplit(
    gsub("\\\\", "/", relative), "/", fixed = TRUE
  )[[1L]]
  traversal <- any(path_components == "..")
  require_gate(
    !traversal,
    paste(label, "is outside the evidence bundle")
  )
  candidate <- if (grepl("^([A-Za-z]:|/)", relative)) {
    relative
  } else {
    file.path(bundle_root, relative)
  }
  path <- normalizePath(candidate, winslash = "/", mustWork = FALSE)
  compared_path <- if (.Platform$OS.type == "windows") tolower(path) else path
  compared_root <- if (.Platform$OS.type == "windows") {
    tolower(bundle_root)
  } else {
    bundle_root
  }
  inside <- startsWith(compared_path, paste0(compared_root, "/"))
  require_gate(inside, paste(label, "is outside the evidence bundle"))
  exists <- file.exists(path)
  require_gate(exists, paste(label, "file is missing"))
  valid_sha <- is_sha256(sha_value)
  require_gate(valid_sha, paste(label, "SHA-256 is invalid"))
  if (exists && valid_sha) {
    actual <- digest::digest(
      path, algo = "sha256", file = TRUE, serialize = FALSE
    )
    require_gate(
      identical(actual, text_value(sha_value)),
      paste(label, "SHA-256 does not match the retained file")
    )
  }
  path
}
read_evidence_json <- function(path, label) {
  value <- tryCatch(
    jsonlite::read_json(path, simplifyVector = FALSE),
    error = identity
  )
  if (inherits(value, "error")) {
    require_gate(FALSE, paste(label, "is not valid JSON"))
    return(NULL)
  }
  value
}
read_evidence_text <- function(path) {
  paste(readLines(path, warn = FALSE), collapse = "\n")
}
require_evidence_text <- function(path, pattern, label) {
  if (!file.exists(path)) return(invisible(FALSE))
  require_gate(
    grepl(pattern, read_evidence_text(path), fixed = TRUE),
    paste(label, "does not contain:", pattern)
  )
  invisible(TRUE)
}

require_gate(
  identical(text_value(manifest$schema), "cudaverse-candidate-evidence/1"),
  "unexpected candidate-evidence schema"
)
commit <- text_value(manifest$candidate$source_commit)
require_gate(is_commit(commit), "candidate source commit is invalid")
version <- text_value(manifest$candidate$version)
branch <- text_value(manifest$candidate$branch)
policy <- candidate_release_policy(version, branch)
require_gate(!is.null(policy),
             "candidate branch/version is not a supported release line")
requires_session_report <- !is.null(policy) &&
  identical(policy$line, "0.4")
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
tarball_path <- check_evidence_file(
  manifest$source_tarball$file,
  manifest$source_tarball$sha256,
  "source tarball"
)
require_gate(
  identical(basename(tarball_path), paste0("cudaverse_", version, ".tar.gz")),
  "source tarball name does not match the candidate version"
)
inspect_source_tarball <- function(path) {
  members <- tryCatch(utils::untar(path, list = TRUE), error = identity)
  if (inherits(members, "error")) {
    require_gate(FALSE, "source tarball is not a readable tar archive")
    return(invisible(FALSE))
  }
  normalized <- gsub("\\\\", "/", members)
  unsafe <- vapply(normalized, function(member) {
    components <- strsplit(member, "/", fixed = TRUE)[[1L]]
    grepl("^([A-Za-z]:|/)", member) || any(components == "..")
  }, logical(1L))
  require_gate(!any(unsafe), "source tarball contains an unsafe member path")
  root <- unique(vapply(
    strsplit(normalized, "/", fixed = TRUE), `[[`, character(1L), 1L
  ))
  require_gate(
    identical(root, "cudaverse") &&
      "cudaverse/DESCRIPTION" %in% normalized,
    "source tarball does not contain one cudaverse package root"
  )
  if (any(unsafe) || !identical(root, "cudaverse") ||
      !"cudaverse/DESCRIPTION" %in% normalized) {
    return(invisible(FALSE))
  }
  extracted <- tempfile("cudaverse-source-tarball-")
  dir.create(extracted)
  on.exit(unlink(extracted, recursive = TRUE, force = TRUE), add = TRUE)
  result <- tryCatch({
    utils::untar(path, exdir = extracted)
    TRUE
  }, error = identity)
  if (inherits(result, "error")) {
    require_gate(FALSE, "source tarball could not be extracted safely")
    return(invisible(FALSE))
  }
  description <- tryCatch(
    read.dcf(file.path(extracted, "cudaverse", "DESCRIPTION")),
    error = identity
  )
  if (inherits(description, "error")) {
    require_gate(FALSE, "source tarball DESCRIPTION is unreadable")
    return(invisible(FALSE))
  }
  require_gate(
    identical(unname(description[[1L, "Package"]]), "cudaverse") &&
      identical(unname(description[[1L, "Version"]]), version),
    "source tarball DESCRIPTION does not match the candidate package/version"
  )
  redistribution <- tryCatch(
    suppressMessages(check_redistributables(
      file.path(extracted, "cudaverse")
    )),
    error = identity
  )
  require_gate(
    !inherits(redistribution, "error"),
    if (inherits(redistribution, "error")) {
      paste("source tarball redistribution gate failed:",
            conditionMessage(redistribution))
    } else {
      "source tarball redistribution gate failed"
    }
  )
  invisible(TRUE)
}
if (file.exists(tarball_path)) inspect_source_tarball(tarball_path)
check_log_path <- check_evidence_file(
  manifest$source_tarball$check_log_file,
  manifest$source_tarball$check_log_sha256,
  "local check log"
)
require_evidence_text(check_log_path, "Status: OK", "local check log")
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
  artifact_path <- check_evidence_file(
    artifact$file, artifact$sha256, paste(name, "artifact")
  )
  require_gate(number(artifact$bytes) > 0,
               paste(name, "artifact size is invalid"))
  if (file.exists(artifact_path)) {
    require_gate(
      identical(as.numeric(file.info(artifact_path)$size),
                number(artifact$bytes)),
      paste(name, "artifact size does not match the retained file")
    )
  }
}

require_gate(
  identical(text_value(manifest$supply_chain$source_commit), commit),
  "supply-chain evidence is from another source commit"
)
sbom_path <- check_evidence_file(
  manifest$supply_chain$sbom_file,
  manifest$supply_chain$sbom_sha256,
  "SBOM"
)
license_path <- check_evidence_file(
  manifest$supply_chain$license_inventory_file,
  manifest$supply_chain$license_inventory_sha256,
  "license inventory"
)
sbom <- if (file.exists(sbom_path)) read_evidence_json(sbom_path, "SBOM")
if (!is.null(sbom)) {
  require_gate(identical(text_value(sbom$bomFormat), "CycloneDX"),
               "SBOM is not CycloneDX")
  require_gate(
    identical(text_value(sbom$metadata$component$name), "cudaverse") &&
      identical(text_value(sbom$metadata$component$version), version),
    "SBOM component does not match the candidate package/version"
  )
}
for (component in c("CUDA Driver", "cuBLAS", "cuSOLVER", "PTX")) {
  require_evidence_text(
    license_path, component,
    paste("license inventory component", component)
  )
}
require_gate(
  identical(number(manifest$supply_chain$bundled_nvidia_runtime_bytes), 0) &&
    identical(number(manifest$supply_chain$bundled_libtorch_bytes), 0),
  "candidate bundles an NVIDIA runtime or LibTorch"
)

require_gate(
  identical(text_value(manifest$rtx$source_commit), commit),
  "RTX evidence is from another source commit"
)
rtx_path <- check_evidence_file(
  manifest$rtx$report_file,
  manifest$rtx$report_sha256,
  "RTX report"
)
consolidation_path <- check_evidence_file(
  manifest$rtx$consolidation_report_file,
  manifest$rtx$consolidation_report_sha256,
  "RTX consolidation report"
)
package_tests_path <- check_evidence_file(
  manifest$rtx$package_test_report_file,
  manifest$rtx$package_test_report_sha256,
  "RTX package-test report"
)
session_path <- NULL
session_file <- text_value(manifest$rtx$session_report_file)
require_gate(
  !requires_session_report || nzchar(session_file),
  "0.4 candidate is missing the RTX independent-session report"
)
if (nzchar(session_file)) {
  session_path <- check_evidence_file(
    manifest$rtx$session_report_file,
    manifest$rtx$session_report_sha256,
    "RTX independent-session report"
  )
}
rtx_report <- if (file.exists(rtx_path)) {
  read_evidence_json(rtx_path, "RTX report")
}
consolidation_report <- if (file.exists(consolidation_path)) {
  read_evidence_json(consolidation_path, "RTX consolidation report")
}
package_test_report <- if (file.exists(package_tests_path)) {
  read_evidence_json(package_tests_path, "RTX package-test report")
}
session_report <- if (!is.null(session_path) && file.exists(session_path)) {
  read_evidence_json(session_path, "RTX independent-session report")
}
if (!is.null(rtx_report)) {
  rtx_failures <- validate_native_candidate_report(
    rtx_report, expected_commit = commit, expected_version = version
  )
  for (failure in rtx_failures) {
    require_gate(FALSE, paste("RTX report:", failure))
  }
  require_gate(
    identical(text_value(rtx_report$schema), text_value(manifest$rtx$schema)),
    "RTX report schema does not match the manifest"
  )
  require_gate(
    identical(text_value(rtx_report$source$cudaverse$commit), commit) &&
      !logical_value(rtx_report$source$cudaverse$tracked_dirty),
    "RTX report source does not match the clean candidate commit"
  )
  require_gate(
    identical(text_value(rtx_report$software$cudaverse), version),
    "RTX report package version does not match the candidate"
  )
  require_gate(logical_value(rtx_report$overall_pass),
               "RTX report does not record overall_pass")
  require_gate(
    identical(
      text_value(rtx_report$inputs$consolidation$sha256),
      text_value(manifest$rtx$consolidation_report_sha256)
    ) && identical(
      text_value(rtx_report$inputs$package_tests$sha256),
      text_value(manifest$rtx$package_test_report_sha256)
    ),
    "RTX report input SHA-256 values do not match the retained inputs"
  )
}
if (!is.null(consolidation_report)) {
  require_gate(
    identical(text_value(consolidation_report$schema),
              "cudaverse-native-consolidation/1") &&
      identical(text_value(
        consolidation_report$source$cudaverse$commit
      ), commit) &&
      !logical_value(
        consolidation_report$source$cudaverse$tracked_dirty
      ) &&
      identical(text_value(
        consolidation_report$software$cudaverse
      ), version) &&
      logical_value(consolidation_report$overall_pass),
    "RTX consolidation report does not match the passing candidate"
  )
}
if (!is.null(package_test_report)) {
  test_counts <- package_test_report$testthat
  require_gate(
    identical(text_value(package_test_report$schema),
              "cudaverse-native-package-tests/1") &&
      identical(text_value(
        package_test_report$source$cudaverse$commit
      ), commit) &&
      !logical_value(package_test_report$source$cudaverse$tracked_dirty) &&
      identical(text_value(
        package_test_report$software$cudaverse
      ), version) &&
      number(test_counts$tests) > 0 && number(test_counts$expectations) > 0 &&
      identical(number(test_counts$failures), 0) &&
      identical(number(test_counts$errors), 0) &&
      identical(number(test_counts$skips), 0) &&
      logical_value(test_counts$native_ready) &&
      logical_value(test_counts$passed) &&
      logical_value(package_test_report$overall_pass),
    "RTX package-test report does not match the passing no-skip candidate"
  )
}
if (!is.null(session_report)) {
  require_gate(
    identical(text_value(session_report$schema),
              "cudaverse-native-session/1") &&
      identical(text_value(manifest$rtx$session_schema),
                "cudaverse-native-session/1"),
    "RTX independent-session report schema is invalid"
  )
  require_gate(logical_value(manifest$rtx$session_passed),
               "RTX independent-session manifest gate did not pass")
  expected_hardware <- if (is.null(package_test_report)) {
    NULL
  } else {
    text_value(package_test_report$hardware$nvidia_smi)
  }
  session_failures <- validate_native_session_report(
    session_report,
    expected_commit = commit,
    expected_version = version,
    expected_hardware = expected_hardware
  )
  for (failure in session_failures) {
    require_gate(FALSE, paste("RTX independent-session report:", failure))
  }
}
require_gate(
  !requires_session_report || !is.null(session_report),
  "0.4 candidate lacks readable independent-session evidence"
)
if (!is.null(consolidation_report) && !is.null(package_test_report)) {
  require_gate(
    identical(
      unlist(
        consolidation_report$hardware$nvidia_smi,
        recursive = TRUE, use.names = FALSE
      ),
      unlist(
        package_test_report$hardware$nvidia_smi,
        recursive = TRUE, use.names = FALSE
      )
    ) && identical(
      text_value(consolidation_report$software$R),
      text_value(package_test_report$software$R)
    ),
    "RTX input reports do not identify the same hardware and R runtime"
  )
}
for (gate in c("parity", "structured_recovery", "interruption",
               "backend_reuse", "no_skips")) {
  require_gate(logical_value(manifest$rtx[[gate]]),
               paste("RTX", gate, "gate did not pass"))
  if (!is.null(rtx_report)) {
    require_gate(
      identical(logical_value(manifest$rtx[[gate]]),
                logical_value(rtx_report$gates[[gate]])),
      paste("RTX", gate, "gate does not match the retained report")
    )
  }
}
for (kind in c("dense", "sparse")) {
  lifecycle <- manifest$rtx$lifecycle[[kind]]
  require_gate(number(lifecycle$cycles) >= 1000,
               paste(kind, "lifecycle has fewer than 1,000 cycles"))
  require_gate(number(lifecycle$post_cleanup_difference_bytes) <= 1024^2,
               paste(kind, "lifecycle exceeds the 1 MiB cleanup ceiling"))
  require_gate(logical_value(lifecycle$no_double_free),
               paste(kind, "lifecycle does not prove no double-free"))
  require_gate(identical(number(
    lifecycle$tracked_current_difference_bytes
  ), 0), paste(kind, "lifecycle retains tracked bytes"))
  require_gate(logical_value(lifecycle$passed),
               paste(kind, "lifecycle did not pass"))
  if (!is.null(rtx_report)) {
    retained <- rtx_report$lifecycle[[kind]]
    require_gate(
      identical(number(lifecycle$cycles), number(retained$cycles)) &&
        identical(
          number(lifecycle$post_cleanup_difference_bytes),
          number(retained$post_cleanup_difference_bytes)
        ) &&
        identical(
          number(lifecycle$tracked_current_difference_bytes),
          number(retained$tracked_current_difference_bytes)
        ) &&
        identical(
          logical_value(lifecycle$no_double_free),
          logical_value(retained$no_double_free)
        ) &&
        identical(
          logical_value(lifecycle$passed), logical_value(retained$passed)
        ),
      paste(kind, "lifecycle does not match the retained RTX report")
    )
  }
}

require_gate(
  identical(text_value(manifest$benchmark$source_commit), commit),
  "benchmark evidence is from another source commit"
)
benchmark_path <- check_evidence_file(
  manifest$benchmark$report_file,
  manifest$benchmark$report_sha256,
  "benchmark report"
)
check_evidence_file(
  manifest$benchmark$summary_file,
  manifest$benchmark$summary_sha256,
  "benchmark summary"
)
benchmark_check_path <- check_evidence_file(
  manifest$benchmark$report_checker_log_file,
  manifest$benchmark$report_checker_log_sha256,
  "benchmark report-checker log"
)
summary_check_path <- check_evidence_file(
  manifest$benchmark$summary_checker_log_file,
  manifest$benchmark$summary_checker_log_sha256,
  "benchmark summary-checker log"
)
require_evidence_text(
  benchmark_check_path,
  "Benchmark report passed all machine-readable gates:",
  "benchmark report-checker log"
)
require_evidence_text(
  summary_check_path,
  "Benchmark summary matches the exact complete report:",
  "benchmark summary-checker log"
)
benchmark_report <- if (file.exists(benchmark_path)) {
  read_evidence_json(benchmark_path, "benchmark report")
}
if (!is.null(benchmark_report)) {
  require_gate(
    identical(text_value(benchmark_report$schema), "cudaverse-benchmark/1") &&
      identical(text_value(benchmark_report$profile), "full") &&
      logical_value(benchmark_report$complete),
    "benchmark JSON is not a complete full-profile report"
  )
  require_gate(
    identical(text_value(benchmark_report$source$commit), commit) &&
      !logical_value(benchmark_report$source$tracked_dirty),
    "benchmark report source does not match the clean candidate commit"
  )
  require_gate(
    identical(text_value(benchmark_report$software$cudaverse), version),
    "benchmark report package version does not match the candidate"
  )
}
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
pkgdown_log_path <- check_evidence_file(
  manifest$documentation$render_log_file,
  manifest$documentation$render_log_sha256,
  "pkgdown render log"
)
require_evidence_text(
  pkgdown_log_path,
  "Public pkgdown pages and documentation boundary passed:",
  "pkgdown render log"
)
require_gate(!logical_value(manifest$documentation$pages_deployed),
             "candidate evidence records an unauthorized Pages deployment")

outcome <- text_value(manifest$decision$outcome)
require_gate(outcome %in% c("release", "defer", "reduce-scope"),
             "candidate decision is missing or invalid")
require_gate(
  identical(text_value(manifest$decision$source_commit), commit),
  "candidate decision is from another source commit"
)
decision_path <- check_evidence_file(
  manifest$decision$report_file,
  manifest$decision$report_sha256,
  "candidate decision report"
)
require_gate(length(manifest$decision$limitations) > 0,
             "candidate decision does not record remaining limitations")
require_gate(!logical_value(manifest$decision$external_release_action_taken),
             "an external release action was taken before approval")
if (file.exists(decision_path)) {
  decision_text <- paste(readLines(decision_path, warn = FALSE), collapse = "\n")
  require_decision_text <- function(value, message) {
    require_gate(grepl(value, decision_text, fixed = TRUE), message)
  }
  require_decision_text(
    paste0("Source commit: `", commit, "`"),
    "candidate decision report source commit does not match"
  )
  require_decision_text(
    paste0("Candidate version: `", version, "`"),
    "candidate decision report version does not match"
  )
  require_decision_text(
    paste0("Outcome: `", outcome, "`"),
    "candidate decision report outcome does not match"
  )
  require_decision_text(
    "External release action taken: `false`",
    "candidate decision report does not stop before external release action"
  )
  limitations <- unlist(
    manifest$decision$limitations, recursive = TRUE, use.names = FALSE
  )
  for (limitation in limitations) {
    require_decision_text(
      as.character(limitation),
      paste("candidate decision report omits limitation:", limitation)
    )
  }
}

if (length(failures)) {
  stop(
    "Candidate evidence failed:\n- ",
    paste(unique(failures), collapse = "\n- "),
    call. = FALSE
  )
}
message("Candidate evidence matches one exact review source: ", commit)
