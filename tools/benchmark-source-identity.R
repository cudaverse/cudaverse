# A retained benchmark uses a package installed from an exact clean commit.
# The installation manifest is created by install-benchmark-candidate.R,
# outside the package and outside the source checkout.
benchmark_git_value <- function(root, args) {
  output <- suppressWarnings(system2(
    "git", c("-C", shQuote(normalizePath(root, mustWork = TRUE)), args),
    stdout = TRUE, stderr = TRUE
  ))
  if (!is.null(attr(output, "status")) || length(output) != 1L) {
    stop("Could not read the benchmark source Git identity.", call. = FALSE)
  }
  output[[1L]]
}

benchmark_source_identity <- function(root = ".") {
  commit <- benchmark_git_value(root, c("rev-parse", "HEAD"))
  tree <- benchmark_git_value(root, c("rev-parse", paste0(commit, "^{tree}")))
  status <- suppressWarnings(system2(
    "git", c("-C", shQuote(normalizePath(root, mustWork = TRUE)),
             "status", "--porcelain", "--untracked-files=all"),
    stdout = TRUE, stderr = TRUE
  ))
  if (!is.null(attr(status, "status")) ||
      !grepl("^[0-9a-f]{40}$", commit) ||
      !grepl("^[0-9a-f]{40}$", tree)) {
    stop("Could not verify the benchmark source Git state.", call. = FALSE)
  }
  list(commit = commit, tree = tree, clean = !length(status))
}

benchmark_installed_files <- function(package_root) {
  root <- normalizePath(package_root, winslash = "/", mustWork = TRUE)
  paths <- list.files(root, recursive = TRUE, all.files = TRUE,
                      full.names = TRUE, no.. = TRUE)
  paths <- paths[!dir.exists(paths)]
  relative <- substring(gsub("\\\\", "/", paths), nchar(root) + 2L)
  order <- order(relative, method = "radix")
  relative <- relative[order]
  paths <- paths[order]
  if (!length(paths) || anyDuplicated(relative)) {
    stop("The installed benchmark package has no unique file inventory.",
         call. = FALSE)
  }
  hashes <- vapply(paths, digest::digest, character(1L),
                   algo = "sha256", file = TRUE)
  names(hashes) <- relative
  hashes
}

benchmark_installed_payload_sha256 <- function(files) {
  if (!length(files) || is.null(names(files)) || anyDuplicated(names(files))) {
    stop("Invalid installed package file inventory.", call. = FALSE)
  }
  lines <- paste(names(files), unname(files), sep = "\t")
  digest::digest(paste(lines, collapse = "\n"), algo = "sha256",
                 serialize = FALSE)
}

benchmark_verify_install_manifest <- function(
    source, package, installed, package_version, manifest_path) {
  result <- list(
    verified = FALSE, reason = NULL,
    source_commit = source$commit, source_tree = source$tree,
    manifest_sha256 = NULL, installed_payload_sha256 = NULL,
    installed_package = NULL
  )
  fail <- function(reason) {
    result$reason <- reason
    result
  }
  if (!source$clean) return(fail("source checkout is not clean"))
  if (!nzchar(manifest_path) || !file.exists(manifest_path)) {
    return(fail("CUDAVERSE_BENCHMARK_INSTALL_MANIFEST is missing"))
  }
  manifest <- tryCatch(jsonlite::read_json(manifest_path, simplifyVector = FALSE),
                       error = function(...) NULL)
  if (is.null(manifest) ||
      !identical(manifest$schema, "cudaverse-benchmark-install/1") ||
      !identical(manifest$package, package)) {
    return(fail("installation manifest is invalid"))
  }
  if (!identical(manifest$source$commit, source$commit) ||
      !identical(manifest$source$tree, source$tree)) {
    return(fail("installed package was built from another source commit"))
  }
  if (length(installed) != 1L || !nzchar(installed)) {
    return(fail("benchmark package is not installed"))
  }
  installed <- normalizePath(installed, winslash = "/", mustWork = TRUE)
  result$installed_package <- installed
  if (!identical(installed, manifest$installed_package)) {
    return(fail("loaded package does not use the manifest installation"))
  }
  if (!identical(package_version, manifest$package_version)) {
    return(fail("installed package version differs from the manifest"))
  }
  expected_files <- manifest$installed_files
  if (!is.list(expected_files) || !length(expected_files) ||
      is.null(names(expected_files)) || anyDuplicated(names(expected_files))) {
    return(fail("installation manifest lacks an installed file inventory"))
  }
  expected_files <- unlist(expected_files, use.names = TRUE)
  if (!all(grepl("^[0-9a-f]{64}$", expected_files))) {
    return(fail("installation manifest has invalid file digests"))
  }
  current_files <- benchmark_installed_files(installed)
  if (!identical(current_files, expected_files)) {
    return(fail("installed package files differ from the verified installation"))
  }
  payload_sha256 <- benchmark_installed_payload_sha256(current_files)
  if (!identical(payload_sha256, manifest$installed_payload_sha256)) {
    return(fail("installed package payload fingerprint differs from manifest"))
  }
  result$manifest_sha256 <- digest::digest(manifest_path, algo = "sha256",
                                           file = TRUE)
  result$installed_payload_sha256 <- payload_sha256
  result$verified <- TRUE
  result
}

benchmark_installed_source_identity <- function(
    source_root = ".", package = "cudaverse",
    manifest_path = Sys.getenv("CUDAVERSE_BENCHMARK_INSTALL_MANIFEST",
                               unset = "")) {
  source <- benchmark_source_identity(source_root)
  installed <- tryCatch(
    getNamespaceInfo(asNamespace(package), "path"),
    error = function(...) ""
  )
  version <- if (length(installed) == 1L && nzchar(installed)) {
    tryCatch(as.character(utils::packageVersion(
      package, lib.loc = dirname(installed)
    )), error = function(...) "")
  } else {
    ""
  }
  benchmark_verify_install_manifest(
    source, package, installed, version, manifest_path
  )
}
