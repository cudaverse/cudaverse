if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Candidate-evidence self-tests require jsonlite and digest.",
       call. = FALSE)
}
sys.source(
  file.path("tools", "candidate-evidence-io.R"),
  envir = environment()
)

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
version <- "0.3.0.9000"
work <- tempfile("cudaverse-candidate-evidence-")
dir.create(work)
evidence_files <- c(
  tarball = "cudaverse_0.3.0.9000.tar.gz",
  check_log = "local-check.log",
  windows_artifact = "cudaverse-windows.zip",
  linux_artifact = "cudaverse-linux.tar.gz",
  sbom = "sbom.json",
  licenses = "licenses.csv",
  rtx = "rtx.json",
  consolidation = "rtx-consolidation.json",
  package_tests = "rtx-package-tests.json",
  benchmark = "benchmark.json",
  summary = "benchmark-summary.md",
  benchmark_check = "benchmark-check.log",
  summary_check = "benchmark-summary-check.log",
  pkgdown = "pkgdown.log",
  decision = "candidate-decision.md"
)
write_fixture <- function(name, value = "synthetic retained evidence") {
  writeBin(charToRaw(value), file.path(work, evidence_files[[name]]))
}
plain_files <- setdiff(
  names(evidence_files),
  c("sbom", "rtx", "consolidation", "package_tests", "benchmark", "decision")
)
for (name in plain_files) write_fixture(name)
write_fixture("check_log", "* checking candidate\nStatus: OK")
write_fixture(
  "licenses",
  paste("CUDA Driver", "cuBLAS", "cuSOLVER", "PTX", sep = "\n")
)
write_fixture(
  "benchmark_check",
  "Benchmark report passed all machine-readable gates: benchmark.json"
)
write_fixture(
  "summary_check",
  "Benchmark summary matches the exact complete report: benchmark-summary.md"
)
write_fixture(
  "pkgdown",
  "Public pkgdown pages and documentation boundary passed: docs"
)
write_sbom <- function(component_version = version) {
  jsonlite::write_json(
    list(
      bomFormat = "CycloneDX",
      metadata = list(component = list(
        name = "cudaverse", version = component_version
      ))
    ),
    file.path(work, evidence_files[["sbom"]]),
    auto_unbox = TRUE, pretty = TRUE
  )
}
lifecycle <- function() list(
  cycles = 1000L,
  post_cleanup_difference_bytes = 0L,
  tracked_current_difference_bytes = 0L,
  no_double_free = TRUE,
  passed = TRUE
)
write_consolidation <- function(source_commit = commit) {
  jsonlite::write_json(
    list(
      schema = "cudaverse-native-consolidation/1",
      source = list(cudaverse = list(
        commit = source_commit, tracked_dirty = FALSE
      )),
      hardware = list(
        nvidia_smi = "NVIDIA RTX 2000 Ada Generation Laptop GPU"
      ),
      software = list(R = R.version.string, cudaverse = version),
      overall_pass = TRUE
    ),
    file.path(work, evidence_files[["consolidation"]]),
    auto_unbox = TRUE, pretty = TRUE
  )
}
write_package_tests <- function(
  source_commit = commit, skips = 0L,
  hardware = "NVIDIA RTX 2000 Ada Generation Laptop GPU"
) {
  jsonlite::write_json(
    list(
      schema = "cudaverse-native-package-tests/1",
      source = list(cudaverse = list(
        commit = source_commit, tracked_dirty = FALSE
      )),
      hardware = list(nvidia_smi = hardware),
      software = list(R = R.version.string, cudaverse = version),
      testthat = list(
        tests = 10L, expectations = 100L, failures = 0L, errors = 0L,
        skips = skips, native_ready = TRUE, passed = skips == 0L
      ),
      overall_pass = skips == 0L
    ),
    file.path(work, evidence_files[["package_tests"]]),
    auto_unbox = TRUE, pretty = TRUE
  )
}
write_rtx <- function(source_commit = commit) {
  jsonlite::write_json(
    list(
      schema = "cudaverse-native-candidate/1",
      source = list(cudaverse = list(
        commit = source_commit, tracked_dirty = FALSE
      )),
      hardware = list(
        nvidia_smi = "NVIDIA RTX 2000 Ada Generation Laptop GPU"
      ),
      software = list(
        R = R.version.string, cudaverse = version,
        native_diagnostics = list(
          available = TRUE, runtime_complete = TRUE,
          self_test = list(passed = TRUE)
        )
      ),
      contracts = list(
        backend = "cudaverse-backend/1", stage = "cudaverse-stage/1"
      ),
      inputs = list(
        consolidation = list(
          schema = "cudaverse-native-consolidation/1",
          sha256 = digest::digest(
            file.path(work, evidence_files[["consolidation"]]),
            algo = "sha256", file = TRUE, serialize = FALSE
          )
        ),
        package_tests = list(
          schema = "cudaverse-native-package-tests/1",
          sha256 = digest::digest(
            file.path(work, evidence_files[["package_tests"]]),
            algo = "sha256", file = TRUE, serialize = FALSE
          )
        )
      ),
      gates = list(
        parity = TRUE, structured_recovery = TRUE,
        interruption = TRUE, backend_reuse = TRUE, no_skips = TRUE
      ),
      lifecycle = list(dense = lifecycle(), sparse = lifecycle()),
      overall_pass = TRUE
    ),
    file.path(work, evidence_files[["rtx"]]),
    auto_unbox = TRUE, pretty = TRUE
  )
}
write_benchmark <- function(source_commit = commit) {
  jsonlite::write_json(
    list(
      schema = "cudaverse-benchmark/1", profile = "full",
      source = list(commit = source_commit, tracked_dirty = FALSE),
      software = list(cudaverse = version), complete = TRUE
    ),
    file.path(work, evidence_files[["benchmark"]]),
    auto_unbox = TRUE, pretty = TRUE
  )
}
write_decision <- function(source_commit = commit, outcome = "defer",
                           limitation = "synthetic limitation") {
  writeLines(
    c(
      "# Synthetic candidate decision",
      "",
      paste0("- Source commit: `", source_commit, "`"),
      paste0("- Candidate version: `", version, "`"),
      paste0("- Outcome: `", outcome, "`"),
      "- External release action taken: `false`",
      "",
      "## Remaining limitations",
      "",
      paste0("- ", limitation)
    ),
    file.path(work, evidence_files[["decision"]]),
    useBytes = TRUE
  )
}
write_sbom()
write_consolidation()
write_package_tests()
write_rtx()
write_benchmark()
write_decision()
sha_for <- function(name) {
  digest::digest(
    file.path(work, evidence_files[[name]]),
    algo = "sha256", file = TRUE, serialize = FALSE
  )
}
check <- function() list(
  source_commit = commit,
  conclusion = "success",
  url = "https://example.invalid/check"
)
artifact <- function(name) list(
  source_commit = commit,
  file = evidence_files[[name]],
  sha256 = sha_for(name),
  bytes = as.numeric(file.info(file.path(work, evidence_files[[name]]))$size)
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
    source_commit = commit, file = evidence_files[["tarball"]],
    sha256 = sha_for("tarball"),
    check_log_file = evidence_files[["check_log"]],
    check_log_sha256 = sha_for("check_log"),
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
  artifacts = list(
    windows = artifact("windows_artifact"),
    linux = artifact("linux_artifact")
  ),
  supply_chain = list(
    source_commit = commit,
    sbom_file = evidence_files[["sbom"]], sbom_sha256 = sha_for("sbom"),
    license_inventory_file = evidence_files[["licenses"]],
    license_inventory_sha256 = sha_for("licenses"),
    bundled_nvidia_runtime_bytes = 0L, bundled_libtorch_bytes = 0L
  ),
  rtx = list(
    source_commit = commit, report_file = evidence_files[["rtx"]],
    report_sha256 = sha_for("rtx"),
    consolidation_report_file = evidence_files[["consolidation"]],
    consolidation_report_sha256 = sha_for("consolidation"),
    package_test_report_file = evidence_files[["package_tests"]],
    package_test_report_sha256 = sha_for("package_tests"),
    schema = "cudaverse-native-candidate/1", parity = TRUE,
    structured_recovery = TRUE, interruption = TRUE, backend_reuse = TRUE,
    no_skips = TRUE,
    lifecycle = list(dense = lifecycle(), sparse = lifecycle())
  ),
  benchmark = list(
    source_commit = commit, report_file = evidence_files[["benchmark"]],
    report_sha256 = sha_for("benchmark"),
    summary_file = evidence_files[["summary"]],
    summary_sha256 = sha_for("summary"),
    report_checker_log_file = evidence_files[["benchmark_check"]],
    report_checker_log_sha256 = sha_for("benchmark_check"),
    summary_checker_log_file = evidence_files[["summary_check"]],
    summary_checker_log_sha256 = sha_for("summary_check"),
    complete = TRUE, report_checker_passed = TRUE,
    summary_checker_passed = TRUE
  ),
  documentation = list(
    source_commit = commit, pkgdown_passed = TRUE, render_reviewed = TRUE,
    render_log_file = evidence_files[["pkgdown"]],
    render_log_sha256 = sha_for("pkgdown"), pages_deployed = FALSE
  ),
  decision = list(
    source_commit = commit, outcome = "defer",
    report_file = evidence_files[["decision"]],
    report_sha256 = sha_for("decision"),
    limitations = list("synthetic limitation"),
    external_release_action_taken = FALSE
  )
)

spec <- manifest
spec$schema <- "cudaverse-candidate-evidence-input/1"
manifest <- build_candidate_evidence_manifest(spec, work)
stopifnot(
  identical(manifest$schema, "cudaverse-candidate-evidence/1"),
  identical(manifest$candidate$source_commit, commit),
  identical(manifest$rtx$report_sha256, sha_for("rtx")),
  identical(manifest$benchmark$report_sha256, sha_for("benchmark")),
  identical(manifest$artifacts$windows$bytes,
            as.numeric(file.info(file.path(
              work, evidence_files[["windows_artifact"]]
            ))$size))
)
bad_spec <- spec
bad_spec$benchmark$report_file <- "../outside.json"
expect_error_message(
  build_candidate_evidence_manifest(bad_spec, work),
  "must be a relative path inside the evidence bundle"
)

path <- file.path(work, "manifest.json")
write_manifest <- function() {
  jsonlite::write_json(
    manifest, path, auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
}
Sys.setenv(CUDAVERSE_CANDIDATE_MANIFEST = path)

write_manifest()
run_checker()

write_fixture("check_log", "Status: 1 WARNING")
manifest$source_tarball$check_log_sha256 <- sha_for("check_log")
write_manifest()
expect_error_message(run_checker(), "local check log does not contain: Status: OK")
write_fixture("check_log", "Status: OK")
manifest$source_tarball$check_log_sha256 <- sha_for("check_log")

write_fixture("benchmark_check", "benchmark checker did not run")
manifest$benchmark$report_checker_log_sha256 <- sha_for("benchmark_check")
write_manifest()
expect_error_message(
  run_checker(), "benchmark report-checker log does not contain"
)
write_fixture(
  "benchmark_check",
  "Benchmark report passed all machine-readable gates: benchmark.json"
)
manifest$benchmark$report_checker_log_sha256 <- sha_for("benchmark_check")

write_fixture("summary_check", "summary checker did not run")
manifest$benchmark$summary_checker_log_sha256 <- sha_for("summary_check")
write_manifest()
expect_error_message(
  run_checker(), "benchmark summary-checker log does not contain"
)
write_fixture(
  "summary_check",
  "Benchmark summary matches the exact complete report: benchmark-summary.md"
)
manifest$benchmark$summary_checker_log_sha256 <- sha_for("summary_check")

write_fixture("pkgdown", "pkgdown build output without boundary check")
manifest$documentation$render_log_sha256 <- sha_for("pkgdown")
write_manifest()
expect_error_message(run_checker(), "pkgdown render log does not contain")
write_fixture(
  "pkgdown",
  "Public pkgdown pages and documentation boundary passed: docs"
)
manifest$documentation$render_log_sha256 <- sha_for("pkgdown")

write_fixture("licenses", "CUDA Driver\ncuBLAS\nPTX")
manifest$supply_chain$license_inventory_sha256 <- sha_for("licenses")
write_manifest()
expect_error_message(
  run_checker(), "license inventory component cuSOLVER does not contain"
)
write_fixture(
  "licenses",
  paste("CUDA Driver", "cuBLAS", "cuSOLVER", "PTX", sep = "\n")
)
manifest$supply_chain$license_inventory_sha256 <- sha_for("licenses")

write_fixture("benchmark", "tampered evidence")
write_manifest()
expect_error_message(run_checker(), "benchmark report SHA-256 does not match")
write_benchmark()

write_benchmark(paste(rep("c", 40L), collapse = ""))
manifest$benchmark$report_sha256 <- sha_for("benchmark")
write_manifest()
expect_error_message(run_checker(), "benchmark report source does not match")
write_benchmark()
manifest$benchmark$report_sha256 <- sha_for("benchmark")

write_rtx(paste(rep("d", 40L), collapse = ""))
manifest$rtx$report_sha256 <- sha_for("rtx")
write_manifest()
expect_error_message(run_checker(), "RTX report source does not match")
write_rtx()
manifest$rtx$report_sha256 <- sha_for("rtx")

write_package_tests(skips = 1L)
manifest$rtx$package_test_report_sha256 <- sha_for("package_tests")
write_manifest()
expect_error_message(
  run_checker(), "RTX report input SHA-256 values do not match"
)
write_package_tests()
manifest$rtx$package_test_report_sha256 <- sha_for("package_tests")

write_package_tests(hardware = "NVIDIA RTX 2000 synthetic different GPU")
manifest$rtx$package_test_report_sha256 <- sha_for("package_tests")
write_rtx()
manifest$rtx$report_sha256 <- sha_for("rtx")
write_manifest()
expect_error_message(
  run_checker(), "RTX input reports do not identify the same hardware"
)
write_package_tests()
manifest$rtx$package_test_report_sha256 <- sha_for("package_tests")
write_rtx()
manifest$rtx$report_sha256 <- sha_for("rtx")

write_sbom("0.2.0.9000")
manifest$supply_chain$sbom_sha256 <- sha_for("sbom")
write_manifest()
expect_error_message(run_checker(), "SBOM component does not match")
write_sbom()
manifest$supply_chain$sbom_sha256 <- sha_for("sbom")

manifest$benchmark$report_file <- "../outside-bundle.json"
write_manifest()
expect_error_message(run_checker(), "benchmark report is outside")
manifest$benchmark$report_file <- file.path(
  "nested", "..", "..", "outside-bundle.json"
)
write_manifest()
expect_error_message(run_checker(), "benchmark report is outside")
if (.Platform$OS.type != "windows") {
  case_sibling <- file.path(dirname(work), toupper(basename(work)))
  if (!identical(case_sibling, work)) {
    dir.create(case_sibling)
    case_sibling_report <- file.path(case_sibling, "benchmark.json")
    stopifnot(file.copy(
      file.path(work, evidence_files[["benchmark"]]),
      case_sibling_report
    ))
    manifest$benchmark$report_file <- case_sibling_report
    manifest$benchmark$report_sha256 <- digest::digest(
      case_sibling_report,
      algo = "sha256", file = TRUE, serialize = FALSE
    )
    write_manifest()
    expect_error_message(run_checker(), "benchmark report is outside")
  }
}
manifest$benchmark$report_file <- evidence_files[["benchmark"]]
manifest$benchmark$report_sha256 <- sha_for("benchmark")

manifest$benchmark$source_commit <- paste(rep("c", 40L), collapse = "")
write_manifest()
expect_error_message(run_checker(), "benchmark evidence is from another")
manifest$benchmark$source_commit <- commit

write_decision(outcome = "release")
manifest$decision$report_sha256 <- sha_for("decision")
write_manifest()
expect_error_message(run_checker(), "decision report outcome does not match")
write_decision(source_commit = paste(rep("e", 40L), collapse = ""))
manifest$decision$report_sha256 <- sha_for("decision")
write_manifest()
expect_error_message(
  run_checker(), "decision report source commit does not match"
)
write_decision(limitation = "different limitation")
manifest$decision$report_sha256 <- sha_for("decision")
write_manifest()
expect_error_message(
  run_checker(), "decision report omits limitation: synthetic limitation"
)
write_decision()
manifest$decision$report_sha256 <- sha_for("decision")

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
