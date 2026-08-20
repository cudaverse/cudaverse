candidate_evidence_scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value)) default else value[[1L]]
}

candidate_evidence_logical <- function(x) {
  isTRUE(as.logical(candidate_evidence_scalar(x, FALSE)))
}

candidate_evidence_bundle_file <- function(root, value, label) {
  relative <- as.character(candidate_evidence_scalar(value, ""))
  components <- strsplit(
    gsub("\\\\", "/", relative), "/", fixed = TRUE
  )[[1L]]
  if (!nzchar(relative) || grepl("^([A-Za-z]:|/)", relative) ||
      any(components == "..")) {
    stop(label, " must be a relative path inside the evidence bundle.",
         call. = FALSE)
  }
  path <- file.path(root, relative)
  if (!file.exists(path)) stop(label, " is missing: ", path, call. = FALSE)
  list(
    file = gsub("\\\\", "/", relative),
    path = path,
    sha256 = digest::digest(
      path, algo = "sha256", file = TRUE, serialize = FALSE
    ),
    bytes = as.numeric(file.info(path)$size)
  )
}

candidate_evidence_read_json <- function(file, label) {
  value <- tryCatch(
    jsonlite::read_json(file$path, simplifyVector = FALSE),
    error = identity
  )
  if (inherits(value, "error")) {
    stop(label, " is not valid JSON: ", conditionMessage(value),
         call. = FALSE)
  }
  value
}

build_candidate_evidence_manifest <- function(spec, bundle_root) {
  if (!identical(
    as.character(candidate_evidence_scalar(spec$schema, "")),
    "cudaverse-candidate-evidence-input/1"
  )) {
    stop("Unexpected candidate evidence input schema.", call. = FALSE)
  }
  root <- normalizePath(bundle_root, winslash = "/", mustWork = TRUE)
  commit <- as.character(candidate_evidence_scalar(
    spec$candidate$source_commit, ""
  ))
  version <- as.character(candidate_evidence_scalar(
    spec$candidate$version, ""
  ))

  source_tarball <- candidate_evidence_bundle_file(
    root, spec$source_tarball$file, "source tarball"
  )
  check_log <- candidate_evidence_bundle_file(
    root, spec$source_tarball$check_log_file, "local check log"
  )
  artifacts <- lapply(c("windows", "linux"), function(name) {
    value <- candidate_evidence_bundle_file(
      root, spec$artifacts[[name]]$file, paste(name, "artifact")
    )
    list(
      source_commit = commit, file = value$file,
      sha256 = value$sha256, bytes = value$bytes
    )
  })
  names(artifacts) <- c("windows", "linux")

  sbom <- candidate_evidence_bundle_file(
    root, spec$supply_chain$sbom_file, "SBOM"
  )
  licenses <- candidate_evidence_bundle_file(
    root, spec$supply_chain$license_inventory_file, "license inventory"
  )
  rtx <- candidate_evidence_bundle_file(
    root, spec$rtx$report_file, "RTX candidate report"
  )
  consolidation <- candidate_evidence_bundle_file(
    root, spec$rtx$consolidation_report_file, "RTX consolidation report"
  )
  package_tests <- candidate_evidence_bundle_file(
    root, spec$rtx$package_test_report_file, "RTX package-test report"
  )
  rtx_report <- candidate_evidence_read_json(rtx, "RTX candidate report")
  session <- NULL
  session_report <- NULL
  session_file <- as.character(candidate_evidence_scalar(
    spec$rtx$session_report_file, ""
  ))
  if (nzchar(session_file)) {
    session <- candidate_evidence_bundle_file(
      root, session_file, "RTX independent-session report"
    )
    session_report <- candidate_evidence_read_json(
      session, "RTX independent-session report"
    )
  }

  benchmark <- candidate_evidence_bundle_file(
    root, spec$benchmark$report_file, "benchmark report"
  )
  benchmark_summary <- candidate_evidence_bundle_file(
    root, spec$benchmark$summary_file, "benchmark summary"
  )
  benchmark_check <- candidate_evidence_bundle_file(
    root, spec$benchmark$report_checker_log_file,
    "benchmark report-checker log"
  )
  summary_check <- candidate_evidence_bundle_file(
    root, spec$benchmark$summary_checker_log_file,
    "benchmark summary-checker log"
  )
  benchmark_report <- candidate_evidence_read_json(
    benchmark, "benchmark report"
  )
  pkgdown <- candidate_evidence_bundle_file(
    root, spec$documentation$render_log_file, "pkgdown render log"
  )
  decision <- candidate_evidence_bundle_file(
    root, spec$decision$report_file, "candidate decision report"
  )

  checks <- lapply(spec$github_checks, function(check) list(
    source_commit = commit,
    conclusion = as.character(candidate_evidence_scalar(
      check$conclusion, ""
    )),
    url = as.character(candidate_evidence_scalar(check$url, ""))
  ))

  rtx_manifest <- list(
    source_commit = commit,
    report_file = rtx$file, report_sha256 = rtx$sha256,
    consolidation_report_file = consolidation$file,
    consolidation_report_sha256 = consolidation$sha256,
    package_test_report_file = package_tests$file,
    package_test_report_sha256 = package_tests$sha256,
    schema = as.character(candidate_evidence_scalar(
      rtx_report$schema, ""
    )),
    parity = candidate_evidence_logical(rtx_report$gates$parity),
    structured_recovery = candidate_evidence_logical(
      rtx_report$gates$structured_recovery
    ),
    interruption = candidate_evidence_logical(
      rtx_report$gates$interruption
    ),
    backend_reuse = candidate_evidence_logical(
      rtx_report$gates$backend_reuse
    ),
    no_skips = candidate_evidence_logical(rtx_report$gates$no_skips),
    lifecycle = rtx_report$lifecycle
  )
  if (!is.null(session)) {
    rtx_manifest$session_report_file <- session$file
    rtx_manifest$session_report_sha256 <- session$sha256
    rtx_manifest$session_schema <- as.character(candidate_evidence_scalar(
      session_report$schema, ""
    ))
    rtx_manifest$session_passed <- candidate_evidence_logical(
      session_report$passed
    )
  }

  list(
    schema = "cudaverse-candidate-evidence/1",
    candidate = list(
      source_commit = commit, version = version,
      branch = as.character(candidate_evidence_scalar(
        spec$candidate$branch, ""
      )),
      clean = candidate_evidence_logical(spec$candidate$clean)
    ),
    frozen_refs = spec$frozen_refs,
    source_tarball = list(
      source_commit = commit, file = source_tarball$file,
      sha256 = source_tarball$sha256,
      check_log_file = check_log$file,
      check_log_sha256 = check_log$sha256,
      check = spec$source_tarball$check
    ),
    github_checks = checks,
    artifacts = artifacts,
    supply_chain = list(
      source_commit = commit,
      sbom_file = sbom$file, sbom_sha256 = sbom$sha256,
      license_inventory_file = licenses$file,
      license_inventory_sha256 = licenses$sha256,
      bundled_nvidia_runtime_bytes = as.numeric(candidate_evidence_scalar(
        spec$supply_chain$bundled_nvidia_runtime_bytes, NA_real_
      )),
      bundled_libtorch_bytes = as.numeric(candidate_evidence_scalar(
        spec$supply_chain$bundled_libtorch_bytes, NA_real_
      ))
    ),
    rtx = rtx_manifest,
    benchmark = list(
      source_commit = commit,
      report_file = benchmark$file, report_sha256 = benchmark$sha256,
      summary_file = benchmark_summary$file,
      summary_sha256 = benchmark_summary$sha256,
      report_checker_log_file = benchmark_check$file,
      report_checker_log_sha256 = benchmark_check$sha256,
      summary_checker_log_file = summary_check$file,
      summary_checker_log_sha256 = summary_check$sha256,
      complete = candidate_evidence_logical(benchmark_report$complete),
      report_checker_passed = candidate_evidence_logical(
        spec$benchmark$report_checker_passed
      ),
      summary_checker_passed = candidate_evidence_logical(
        spec$benchmark$summary_checker_passed
      )
    ),
    documentation = list(
      source_commit = commit,
      pkgdown_passed = candidate_evidence_logical(
        spec$documentation$pkgdown_passed
      ),
      render_reviewed = candidate_evidence_logical(
        spec$documentation$render_reviewed
      ),
      render_log_file = pkgdown$file,
      render_log_sha256 = pkgdown$sha256,
      pages_deployed = candidate_evidence_logical(
        spec$documentation$pages_deployed
      )
    ),
    decision = list(
      source_commit = commit,
      outcome = as.character(candidate_evidence_scalar(
        spec$decision$outcome, ""
      )),
      report_file = decision$file, report_sha256 = decision$sha256,
      limitations = spec$decision$limitations,
      external_release_action_taken = candidate_evidence_logical(
        spec$decision$external_release_action_taken
      )
    )
  )
}
