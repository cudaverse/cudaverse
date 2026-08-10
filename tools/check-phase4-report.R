if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking the Phase 4 report requires `jsonlite`.")
}

input <- Sys.getenv(
  "CUDAVERSE_PHASE4_REPORT",
  unset = file.path("inst", "reports", "native", "phase4-rtx2000.json")
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)

scalar <- function(x) unlist(x, recursive = TRUE, use.names = FALSE)[[1L]]
number <- function(x) as.numeric(scalar(x))
logical_value <- function(x) isTRUE(as.logical(scalar(x)))
characters <- function(x) {
  as.character(unlist(x, recursive = TRUE, use.names = FALSE))
}

failures <- character()
require_gate <- function(value, message) {
  if (!isTRUE(value)) failures <<- c(failures, message)
}

require_gate(
  identical(scalar(report$schema), "cudaverse-native-phase4/1"),
  "unexpected report schema"
)
require_gate(
  grepl("RTX 2000", paste(characters(report$hardware$nvidia_smi),
                           collapse = " "), fixed = TRUE),
  "report was not generated on the RTX 2000"
)
require_gate(
  identical(scalar(report$software$cudaverse), "0.2.0.9000"),
  "unexpected cudaverse version"
)
require_gate(
  identical(scalar(report$software$cudaverseCUDA), "0.4.0.9000"),
  "unexpected cudaverseCUDA version"
)
require_gate(
  !logical_value(report$source$cudaverse$tracked_dirty),
  "cudaverse source had tracked changes during the benchmark"
)
require_gate(
  !logical_value(report$source$cudaverseCUDA$tracked_dirty),
  "cudaverseCUDA source had tracked changes during the benchmark"
)
require_gate(
  nchar(scalar(report$source$cudaverse$commit)) == 40L &&
    nchar(scalar(report$source$cudaverseCUDA$commit)) == 40L,
  "source commits were not recorded"
)
require_gate(
  number(report$contract$warmup_runs) >= 2L &&
    number(report$contract$timed_runs) >= 5L,
  "benchmark requires at least two warm-ups and five timed runs"
)
require_gate(
  identical(
    scalar(report$contract$native_selection),
    paste(
      "automatic; contract, capabilities, runtime components, and cached",
      "self-test passed"
    )
  ),
  "native selection policy changed unexpectedly"
)
require_gate(
  identical(sort(characters(report$contract$tensor_dtypes)),
            c("float32", "float64")) &&
    identical(scalar(report$contract$sparse_pipeline_dtype), "float64"),
  "dtype contract changed unexpectedly"
)

required_backends <- c("base", "native", "torch")
required_cases <- c("1000x50@0.1", "5000x100@0.03", "10000x128@0.01")
require_gate(
  identical(sort(names(report$benchmarks)), sort(required_cases)),
  "required sparse benchmark cases are missing"
)

for (case_name in required_cases) {
  case <- report$benchmarks[[case_name]]
  require_gate(!is.null(case), paste(case_name, "benchmark is missing"))
  if (is.null(case)) next
  require_gate(
    identical(sort(names(case)), sort(required_backends)),
    paste(case_name, "does not contain base/native/torch")
  )
  for (backend in required_backends) {
    result <- case[[backend]]
    label <- paste(case_name, backend)
    require_gate(!is.null(result), paste(label, "result is missing"))
    if (is.null(result)) next
    require_gate(
      identical(scalar(result$status), "complete"),
      paste(label, "did not complete")
    )
    require_gate(
      logical_value(result$validation$passed),
      paste(label, "failed numerical parity")
    )
    require_gate(
      length(characters(result$full_pipeline$runs_seconds)) >= 5L,
      paste(label, "has too few timed runs")
    )
    require_gate(
      number(result$validation$normalized_max_relative_error) <= 1e-10,
      paste(label, "normalization exceeded tolerance")
    )
    require_gate(
      number(result$validation$pca_effective_rank) >= 1L &&
        number(result$validation$pca_effective_rank) <=
          number(result$n_components),
      paste(label, "has an invalid numerical PCA rank")
    )
    require_gate(
      number(result$validation$pca_subspace_projector_max_absolute_error) <=
        1e-8,
      paste(label, "identifiable PCA subspace exceeded tolerance")
    )
    require_gate(
      number(result$validation$pca_reconstruction_max_relative_error) <= 1e-8,
      paste(label, "PCA reconstruction exceeded tolerance")
    )
    require_gate(
      logical_value(result$validation$knn_indices_identical) &&
        number(result$validation$knn_distance_max_relative_error) <= 1e-8,
      paste(label, "stable exact kNN parity failed")
    )

    if (identical(backend, "native")) {
      normalized_stages <- characters(
        result$provenance$normalized$stages$stage
      )
      pca_stages <- characters(result$provenance$pca$stages$stage)
      knn_stages <- characters(result$provenance$knn$stages$stage)
      require_gate(
        identical(scalar(result$provenance$normalized$schema),
                  "cudaverse-stage/1") &&
          identical(scalar(result$provenance$pca$schema),
                    "cudaverse-stage/1") &&
          identical(scalar(result$provenance$knn$schema),
                    "cudaverse-stage/1"),
        paste(label, "changed the provenance schema")
      )
      require_gate(
        "normalization" %in% normalized_stages &&
          all(c(
            "normalization", "sparse_to_dense", "preprocessing",
            "decomposition", "scores_resident"
          ) %in% pca_stages) &&
          all(c("distance", "neighbor_selection") %in% knn_stages),
        paste(label, "is missing resident pipeline stages")
      )
      require_gate(
        logical_value(
          result$provenance$sparse_storage_device_resident
        ) && logical_value(result$provenance$pca_scores_device_resident),
        paste(label, "did not retain sparse/PCA state on the device")
      )
      require_gate(
        number(
          result$memory$native_tracker_post_cleanup_difference_bytes
        ) == 0,
        paste(label, "left tracked native allocations after cleanup")
      )
    }
  }
}

hardware <- report$hardware_validation
require_gate(
  identical(scalar(hardware$schema),
            "cudaverse-native-phase4-validation/1"),
  "unexpected hardware-validation schema"
)
for (gate in c(
  "auto_selection", "dtype_surface", "dense_lifecycle", "lifecycle",
  "shared_ownership", "structured_error",
  "injected_cuda_error", "interruption", "resident_pipeline",
  "stable_ties"
)) {
  require_gate(
    logical_value(hardware[[gate]]$passed),
    paste("hardware gate failed:", gate)
  )
}
require_gate(
  identical(scalar(hardware$auto_selection$selected_backend), "native") &&
    "native" %in% characters(
      hardware$auto_selection$auto_eligible_backends
    ) &&
    identical(
      scalar(hardware$auto_selection$selection_reason),
      "native_auto_eligible"
    ) &&
    identical(
      scalar(hardware$auto_selection$contract_schema),
      "cudaverse-backend/1"
    ) &&
    logical_value(hardware$auto_selection$capability_compatible) &&
    logical_value(hardware$auto_selection$runtime_complete) &&
    logical_value(hardware$auto_selection$self_test$passed),
  "native automatic-selection evidence is incomplete"
)
for (dtype in c("float32", "float64")) {
  surface <- hardware$dtype_surface$dtypes[[dtype]]
  require_gate(
    logical_value(surface$passed) &&
      logical_value(surface$all_outputs_native),
    paste(dtype, "tensor surface failed parity or left native CUDA")
  )
}
require_gate(
  number(hardware$dense_lifecycle$cycles) == 1000L &&
    number(
      hardware$dense_lifecycle$whole_device_absolute_difference_bytes
    ) <= 1024^2 &&
    number(hardware$dense_lifecycle$tracked_current_difference_bytes) == 0,
  "1,000-cycle dense VRAM lifecycle contract failed"
)
require_gate(
  number(hardware$lifecycle$cycles) == 1000L &&
    number(hardware$lifecycle$whole_device_absolute_difference_bytes) <=
      1024^2 &&
    number(hardware$lifecycle$tracked_current_difference_bytes) == 0,
  "1,000-cycle sparse VRAM lifecycle contract failed"
)
require_gate(
  identical(scalar(hardware$structured_error$backend), "native") &&
    identical(
      scalar(hardware$structured_error$operation),
      "sparse_matmul_dense"
    ),
  "sparse errors were not translated into the native condition contract"
)
require_gate(
  logical_value(hardware$interruption$interrupted) &&
    logical_value(hardware$interruption$backend_reusable),
  "interruption did not leave the backend reusable"
)
require_gate(
  logical_value(hardware$stable_ties$indices_identical) &&
    number(hardware$stable_ties$distance_max_absolute_error) <= 1e-10,
  "equal-distance kNN results were not stably ordered by source row"
)

regression <- report$benchmark_regression
require_gate(
  identical(
    scalar(regression$schema),
    "cudaverse-native-phase4-regression/1"
  ),
  "unexpected benchmark-regression schema"
)
require_gate(
  identical(scalar(regression$baseline_schema),
            "cudaverse-native-phase3/1") &&
    grepl("^[0-9a-f]{64}$", scalar(regression$baseline_sha256)),
  "Phase 3 baseline identity is missing"
)
for (case_name in required_cases) {
  candidate <- regression$cases[[case_name]]
  require_gate(
    logical_value(candidate$passed) &&
      logical_value(candidate$median_passed) &&
      logical_value(candidate$p95_passed) &&
      logical_value(candidate$peak_passed) &&
      number(candidate$candidate_median_seconds) <=
        number(candidate$median_limit_seconds) &&
      number(candidate$candidate_p95_seconds) <=
        number(candidate$p95_limit_seconds) &&
      number(candidate$candidate_peak_bytes) <=
        number(candidate$peak_limit_bytes),
    paste(case_name, "regressed beyond the Phase 3 performance envelope")
  )
}
for (package in c("cudaverse", "cudaverseCUDA")) {
  size <- regression$installed_size[[package]]
  require_gate(
    logical_value(size$passed) &&
      number(size$candidate_bytes) <= number(size$limit_bytes),
    paste(package, "installed size regressed beyond the Phase 3 envelope")
  )
}
require_gate(
  logical_value(regression$passed),
  "benchmark-regression gate failed"
)
require_gate(logical_value(report$overall_pass), "overall report gate failed")

if (length(failures)) {
  stop(
    "Phase 4 report check failed:\n- ",
    paste(unique(failures), collapse = "\n- ")
  )
}
message("Phase 4 report passed all machine-readable gates: ", input)
