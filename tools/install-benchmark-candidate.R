if (!requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Benchmark candidate installation requires digest and jsonlite.",
       call. = FALSE)
}
sys.source(file.path("tools", "benchmark-source-identity.R"),
           envir = environment())

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L || !nzchar(arguments[[1L]])) {
  stop("Usage: Rscript tools/install-benchmark-candidate.R <new-output-directory>",
       call. = FALSE)
}
source_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
source <- benchmark_source_identity(source_root)
if (!source$clean) {
  stop("Benchmark installation requires a clean committed checkout.",
       call. = FALSE)
}
output_parent <- normalizePath(dirname(arguments[[1L]]),
                               winslash = "/", mustWork = TRUE)
output <- paste0(output_parent, "/", basename(arguments[[1L]]))
if (file.exists(output) || dir.exists(output)) {
  stop("The benchmark installation output directory must be new.",
       call. = FALSE)
}
if (startsWith(paste0(output, "/"), paste0(source_root, "/"))) {
  stop("Place the benchmark installation outside the source checkout.",
       call. = FALSE)
}
dir.create(output, recursive = FALSE)
archive <- file.path(output, "source.tar")
archive_output <- suppressWarnings(system2(
  "git",
  c("-C", shQuote(source_root), "archive", "--format=tar",
    paste0("--output=", shQuote(archive)), source$commit),
  stdout = TRUE, stderr = TRUE
))
if (!is.null(attr(archive_output, "status")) || !file.exists(archive)) {
  stop("Could not archive the exact source commit: ",
       paste(archive_output, collapse = "\n"), call. = FALSE)
}
source_copy <- file.path(output, "source")
dir.create(source_copy)
utils::untar(archive, exdir = source_copy)
if (!file.exists(file.path(source_copy, "DESCRIPTION"))) {
  stop("The archived benchmark source lacks DESCRIPTION.", call. = FALSE)
}
library <- file.path(output, "lib")
dir.create(library)
install_log <- file.path(output, "install.log")
r_command <- file.path(R.home("bin"), "R")
install_output <- suppressWarnings(system2(
  r_command,
  c("CMD", "INSTALL", "--preclean", paste0("--library=", shQuote(library)),
    shQuote(source_copy)),
  stdout = TRUE, stderr = TRUE
))
writeLines(install_output, install_log, useBytes = TRUE)
if (!is.null(attr(install_output, "status"))) {
  stop("R CMD INSTALL failed; see ", install_log, call. = FALSE)
}
installed <- file.path(library, "cudaverse")
if (!dir.exists(installed)) {
  stop("R CMD INSTALL did not produce the isolated cudaverse package.",
       call. = FALSE)
}
source_after_install <- benchmark_source_identity(source_root)
if (!source_after_install$clean ||
    !identical(source_after_install$commit, source$commit) ||
    !identical(source_after_install$tree, source$tree)) {
  stop("Source checkout changed during benchmark candidate installation.",
       call. = FALSE)
}
installed <- normalizePath(installed, winslash = "/", mustWork = TRUE)
files <- benchmark_installed_files(installed)
manifest <- list(
  schema = "cudaverse-benchmark-install/1",
  package = "cudaverse",
  package_version = as.character(utils::packageVersion("cudaverse", lib.loc = library)),
  source = list(commit = source$commit, tree = source$tree),
  source_archive_sha256 = digest::digest(archive, algo = "sha256", file = TRUE),
  installed_package = installed,
  installed_payload_sha256 = benchmark_installed_payload_sha256(files),
  installed_files = as.list(files)
)
manifest_path <- file.path(output, "manifest.json")
jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE,
                     pretty = TRUE, null = "null")
if (!file.exists(manifest_path)) {
  stop("Could not retain the benchmark installation manifest.",
       call. = FALSE)
}
message("Verified candidate installation: ", installed)
message("Manifest: ", manifest_path)
message("Set R_LIBS_USER to ", normalizePath(library, winslash = "/"),
        " and CUDAVERSE_BENCHMARK_INSTALL_MANIFEST to ",
        normalizePath(manifest_path, winslash = "/"),
        " in the benchmark process.")
