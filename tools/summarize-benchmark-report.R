if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop(
    "Summarizing benchmark reports requires jsonlite and digest.",
    call. = FALSE
  )
}

input <- Sys.getenv("CUDAVERSE_BENCHMARK_REPORT", unset = "")
output <- Sys.getenv("CUDAVERSE_BENCHMARK_SUMMARY", unset = "")
allow_incomplete <- identical(
  tolower(Sys.getenv(
    "CUDAVERSE_BENCHMARK_ALLOW_INCOMPLETE", unset = "false"
  )),
  "true"
)
if (!nzchar(input) || !file.exists(input)) {
  stop("Set CUDAVERSE_BENCHMARK_REPORT to an existing report.", call. = FALSE)
}
if (!nzchar(output)) {
  stop("Set CUDAVERSE_BENCHMARK_SUMMARY to the output Markdown path.",
       call. = FALSE)
}

sha256_before_read <- digest::digest(
  input, algo = "sha256", file = TRUE, serialize = FALSE
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)
sha256_after_read <- digest::digest(
  input, algo = "sha256", file = TRUE, serialize = FALSE
)
if (!identical(sha256_before_read, sha256_after_read)) {
  stop("The benchmark report changed while it was being read.", call. = FALSE)
}
scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value)) default else value[[1L]]
}
number <- function(x) suppressWarnings(as.numeric(scalar(x, NA_real_)))
text_value <- function(x, default = "n/a") {
  value <- scalar(x, default)
  if (is.na(value) || !nzchar(as.character(value))) default else
    as.character(value)
}
logical_value <- function(x) isTRUE(as.logical(scalar(x, FALSE)))
format_seconds <- function(x) {
  value <- number(x)
  if (!is.finite(value)) "n/a" else formatC(value, digits = 6L, format = "f")
}
format_number <- function(x, digits = 3L) {
  value <- number(x)
  if (!is.finite(value)) "n/a" else trimws(formatC(
    value, digits = digits, format = "g"
  ))
}
format_bytes <- function(x) {
  value <- number(x)
  if (!is.finite(value)) "n/a" else
    format(round(value), big.mark = ",", scientific = FALSE, trim = TRUE)
}
format_mib <- function(x) {
  value <- number(x)
  if (!is.finite(value)) "n/a" else
    formatC(value / 1024^2, digits = 2L, format = "f")
}
escape_markdown <- function(x) gsub("\\|", "\\\\|", as.character(x))
append_line <- function(...) lines <<- c(lines, paste0(...))

if (!identical(text_value(report$schema), "cudaverse-benchmark/1")) {
  stop("Unexpected benchmark report schema.", call. = FALSE)
}
is_complete <- logical_value(report$complete)
if (!is_complete && !allow_incomplete) {
  stop(
    paste(
      "The report is incomplete. Set",
      "CUDAVERSE_BENCHMARK_ALLOW_INCOMPLETE=true only for a draft summary."
    ),
    call. = FALSE
  )
}

report_sha256 <- sha256_after_read
commit <- text_value(report$source$commit)
backends <- unlist(
  report$contract$backends, recursive = TRUE, use.names = FALSE
)
lines <- character()

append_line("# CP-06 full benchmark evidence")
append_line("")
if (!is_complete) {
  append_line("> **DRAFT -- incomplete report.** Only atomically completed ",
              "case/backend results are shown.")
  append_line("")
}
append_line("- Schema: `", text_value(report$schema), "`")
append_line("- Profile: `", text_value(report$profile), "`")
append_line("- Source commit: `", commit, "`")
append_line("- Source tracked dirty: `",
            tolower(as.character(logical_value(report$source$tracked_dirty))),
            "`")
append_line("- Report complete: `", tolower(as.character(is_complete)), "`")
append_line("- Report SHA-256: `", report_sha256, "`")
append_line("- Report generated: ", text_value(report$generated_at_utc))
append_line("- Hardware: ", escape_markdown(
  text_value(report$hardware$nvidia_smi)
))
append_line("- R: ", text_value(report$software$R))
append_line("- cudaverse: `", text_value(report$software$cudaverse), "`")
append_line("- torch: `", text_value(report$software$torch), "`")
append_line(
  "- Stage sampling: ",
  text_value(
    report$contract$stage_sampling,
    "legacy/unspecified; inspect the runner at the recorded source commit"
  )
)
append_line(
  "- Memory sampling: ",
  text_value(
    report$contract$memory_sampling,
    "legacy/unspecified; inspect the runner at the recorded source commit"
  )
)
append_line("")
append_line("## Installed footprint")
append_line("")
append_line("| Component | Installed bytes |")
append_line("|---|---:|")
append_line("| cudaverse | ",
            format_bytes(report$installed_size_bytes$cudaverse), " |")
append_line("| optional torch | ",
            format_bytes(report$installed_size_bytes$torch), " |")
append_line("| CUDA runtime bundled by cudaverse | ",
            format_bytes(report$installed_size_bytes$bundled_cuda_runtime),
            " |")
append_line("")

matmul_ids <- names(report$cases)[vapply(
  report$cases,
  function(x) identical(text_value(x$definition$family), "matmul"),
  logical(1)
)]
append_line("## Matrix multiplication")
append_line("")
append_line(paste(
  "| Case | Backend | Host median (s) | Host p95 (s) |",
  "Resident median (s) | Resident p95 (s) | Peak MiB |",
  "Max relative error | Parity |"
))
append_line("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for (case_id in matmul_ids) {
  values <- report$cases[[case_id]]$backends
  for (backend in intersect(backends, names(values))) {
    value <- values[[backend]]
    append_line(
      "| ", escape_markdown(case_id), " | ", backend, " | ",
      format_seconds(value$warm$host_boundary$median_seconds), " | ",
      format_seconds(value$warm$host_boundary$p95_seconds), " | ",
      format_seconds(value$warm$resident_compute$median_seconds), " | ",
      format_seconds(value$warm$resident_compute$p95_seconds), " | ",
      format_mib(value$memory$backend_allocator_peak_bytes), " | ",
      format_number(value$validation$max_relative_error), " | ",
      if (logical_value(value$validation$passed)) "pass" else "fail", " |"
    )
  }
}
append_line("")

pipeline_ids <- setdiff(names(report$cases), matmul_ids)
append_line("## PCA and exact kNN pipelines")
append_line("")
append_line(paste(
  "| Case | Backend | Host median (s) | Host p95 (s) |",
  "Resident continuation (s) | PCA stage (s) | kNN stage (s) |",
  "Peak MiB | Validation |"
))
append_line("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for (case_id in pipeline_ids) {
  values <- report$cases[[case_id]]$backends
  for (backend in intersect(backends, names(values))) {
    value <- values[[backend]]
    resident <- value$warm$resident_continuation
    resident_text <- if (!is.null(resident$median_seconds)) {
      format_seconds(resident$median_seconds)
    } else {
      text_value(resident$status, "not separable")
    }
    validation <- if (identical(
      text_value(value$validation$reference_backend, ""), "self"
    )) {
      "self-reference; pass"
    } else {
      paste0(
        "projector ",
        format_number(value$validation$pca_projector_max_absolute_error),
        "; reconstruction ",
        format_number(value$validation$pca_reconstruction_max_relative_error),
        "; kNN distance ",
        format_number(value$validation$knn_distance_max_relative_error),
        "; indices ",
        if (logical_value(value$validation$knn_indices_identical)) {
          "exact"
        } else {
          "different"
        },
        "; ", if (logical_value(value$validation$passed)) "pass" else "fail"
      )
    }
    append_line(
      "| ", escape_markdown(case_id), " | ", backend, " | ",
      format_seconds(value$warm$host_boundary$median_seconds), " | ",
      format_seconds(value$warm$host_boundary$p95_seconds), " | ",
      resident_text, " | ",
      format_seconds(value$warm$stages$pca$median_seconds), " | ",
      format_seconds(value$warm$stages$knn$median_seconds), " | ",
      format_mib(value$memory$backend_allocator_peak_bytes), " | ",
      validation, " |"
    )
  }
}
append_line("")

comparison <- function(candidate_name, candidate, reference_name, reference) {
  if (!is.finite(candidate) || !is.finite(reference) || candidate <= 0 ||
      reference <= 0) {
    return("comparison unavailable")
  }
  if (candidate <= reference) {
    sprintf(
      "%s median was %.1f%% lower than %s (ratio %.2fx)",
      candidate_name, 100 * (1 - candidate / reference), reference_name,
      reference / candidate
    )
  } else {
    sprintf(
      "%s median was %.1f%% higher than %s (ratio %.2fx)",
      candidate_name, 100 * (candidate / reference - 1), reference_name,
      candidate / reference
    )
  }
}
append_line("## Workload-specific observations")
append_line("")
for (case_id in names(report$cases)) {
  values <- report$cases[[case_id]]$backends
  if (!all(c("base", "native", "torch") %in% names(values))) next
  host <- vapply(c("base", "native", "torch"), function(backend) {
    number(values[[backend]]$warm$host_boundary$median_seconds)
  }, numeric(1))
  fastest <- names(host)[which.min(host)]
  append_line(
    "- `", case_id, "`: ", fastest,
    " had the numerically lowest host-boundary median; ",
    comparison("native", host[["native"]], "base", host[["base"]]),
    ", and ",
    comparison("native", host[["native"]], "torch", host[["torch"]]),
    "."
  )
  if (identical(
    text_value(report$cases[[case_id]]$definition$family), "matmul"
  )) {
    native_resident <- number(
      values$native$warm$resident_compute$median_seconds
    )
    torch_resident <- number(
      values$torch$warm$resident_compute$median_seconds
    )
    append_line(
      "  Resident matmul: ",
      comparison("native", native_resident, "torch", torch_resident), "."
    )
  }
}
append_line("")
append_line("## Interpretation boundaries")
append_line("")
append_line(paste(
  "- These measurements describe one exact source commit on one RTX 2000",
  "Ada system. They do not support a universal GPU speed claim."
))
append_line(paste(
  "- Ratios compare ten-run sample medians descriptively. They are not",
  "confidence intervals or statistical significance tests; small differences",
  "may be measurement noise."
))
append_line(paste(
  "- Host-boundary and resident timings answer different questions.",
  "Dense PCA upload is internal to the public boundary and is therefore",
  "reported as not separable."
))
append_line(paste(
  "- Peak memory is reported with its backend-specific allocator source in",
  "the machine-readable report; torch uses a session high-water source when",
  "its R API cannot reset a peak counter."
))
append_line(paste(
  "- Pipeline base results establish the numerical reference. Native and",
  "torch must match PCA projector/reconstruction, exact kNN indices, and",
  "distance tolerances before their timings are accepted."
))

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
writeLines(lines, output, useBytes = TRUE)
message(
  if (is_complete) "Final" else "Draft",
  " benchmark summary written: ", normalizePath(output, winslash = "/")
)
