if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("The SBOM gate requires the `jsonlite` and `digest` packages.")
}

description <- read.dcf("DESCRIPTION")[1L, ]
sbom <- jsonlite::read_json(
  file.path("inst", "reports", "sbom.cdx.json"), simplifyVector = FALSE
)
if (!identical(sbom$metadata$component$version, unname(description[["Version"]]))) {
  stop("The SBOM component version does not match DESCRIPTION.")
}

component_names <- vapply(sbom$components, `[[`, "", "name")
required <- c(
  "NVIDIA CUDA Driver API", "NVIDIA cuBLAS 12", "NVIDIA cuSOLVER 11",
  "NVIDIA CUDA Compiler",
  "cudaverse_dense_kernels.ptx"
)
missing <- setdiff(required, component_names)
if (length(missing)) {
  stop("The SBOM is missing component(s): ", paste(missing, collapse = ", "))
}

ptx <- sbom$components[[match(
  "cudaverse_dense_kernels.ptx", component_names
)]]
declared_hash <- ptx$hashes[[1L]]$content
actual_hash <- digest::digest(
  file.path("inst", "kernels", "cudaverse_dense_kernels.ptx"),
  algo = "sha256", file = TRUE, serialize = FALSE
)
if (!identical(tolower(declared_hash), tolower(actual_hash))) {
  stop("The SBOM PTX SHA-256 does not match the committed PTX.")
}

license_text <- paste(
  readLines(file.path("inst", "reports", "THIRD_PARTY_LICENSES.md"),
            warn = FALSE),
  collapse = "\n"
)
for (component in c("CUDA Driver", "cuBLAS 12", "cuSOLVER 11", "PTX")) {
  if (!grepl(component, license_text, fixed = TRUE)) {
    stop("The third-party inventory is missing `", component, "`.")
  }
}
message("SBOM gate passed: versions, components, PTX SHA-256, and inventory agree.")
