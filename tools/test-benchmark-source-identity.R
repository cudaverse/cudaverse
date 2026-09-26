if (!requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Benchmark source-identity self-test requires digest and jsonlite.",
       call. = FALSE)
}
sys.source(file.path("tools", "benchmark-source-identity.R"),
           envir = environment())

work <- tempfile("cudaverse-benchmark-identity-")
package_root <- file.path(work, "lib", "cudaverse")
dir.create(file.path(package_root, "R"), recursive = TRUE)
dir.create(file.path(package_root, "libs", "x64"), recursive = TRUE)
writeLines("R code from this candidate", file.path(package_root, "R", "cudaverse.rdb"))
writeBin(as.raw(c(1, 2, 3, 4)),
         file.path(package_root, "libs", "x64", "cudaverse.dll"))
package_root <- normalizePath(package_root, winslash = "/")
manifest_path <- file.path(work, "manifest.json")
commit <- paste(rep("a", 40L), collapse = "")
tree <- paste(rep("b", 40L), collapse = "")
source <- list(commit = commit, tree = tree, clean = TRUE)
files <- benchmark_installed_files(package_root)
manifest <- list(
  schema = "cudaverse-benchmark-install/1",
  package = "cudaverse", package_version = "0.4.1.9000",
  source = list(commit = commit, tree = tree),
  installed_package = package_root,
  installed_payload_sha256 = benchmark_installed_payload_sha256(files),
  installed_files = as.list(files)
)
write_manifest <- function(value = manifest) {
  jsonlite::write_json(value, manifest_path, auto_unbox = TRUE,
                       pretty = TRUE, null = "null")
}
verify <- function(source_value = source, package_path = package_root,
                   version = "0.4.1.9000", manifest_file = manifest_path) {
  benchmark_verify_install_manifest(
    source_value, "cudaverse", package_path, version, manifest_file
  )
}
reject <- function(value, reason) {
  stopifnot(!isTRUE(value$verified), grepl(reason, value$reason, fixed = TRUE))
}
write_manifest()
valid <- verify()
stopifnot(
  isTRUE(valid$verified), identical(valid$source_commit, commit),
  identical(valid$source_tree, tree),
  identical(valid$installed_payload_sha256, manifest$installed_payload_sha256),
  grepl("^[0-9a-f]{64}$", valid$manifest_sha256)
)

stale <- source
stale$commit <- paste(rep("c", 40L), collapse = "")
reject(verify(stale), "another source commit")
stale <- source
stale$tree <- paste(rep("c", 40L), collapse = "")
reject(verify(stale), "another source commit")
stale <- source
stale$clean <- FALSE
reject(verify(stale), "not clean")
reject(verify(manifest_file = file.path(work, "absent.json")), "is missing")
reject(verify(version = "0.4.2"), "version differs")

another_package <- file.path(work, "other", "cudaverse")
dir.create(another_package, recursive = TRUE)
reject(verify(package_path = another_package), "does not use the manifest")

# Identical package versions do not excuse changed R bytecode or a stale DLL.
writeLines("stale R bytecode", file.path(package_root, "R", "cudaverse.rdb"))
reject(verify(), "installed package files differ")
writeLines("R code from this candidate", file.path(package_root, "R", "cudaverse.rdb"))
writeBin(as.raw(c(9, 8, 7, 6)),
         file.path(package_root, "libs", "x64", "cudaverse.dll"))
reject(verify(), "installed package files differ")
writeBin(as.raw(c(1, 2, 3, 4)),
         file.path(package_root, "libs", "x64", "cudaverse.dll"))

changed <- manifest
changed$installed_files[[1L]] <- paste(rep("0", 64L), collapse = "")
write_manifest(changed)
reject(verify(), "installed package files differ")
write_manifest()
stopifnot(isTRUE(verify()$verified))
message("Exact-source benchmark installation identity self-tests passed.")
