if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Building candidate evidence requires jsonlite and digest.",
       call. = FALSE)
}
sys.source(
  file.path("tools", "candidate-evidence-io.R"),
  envir = environment()
)

input <- Sys.getenv("CUDAVERSE_CANDIDATE_INPUT", unset = "")
output <- Sys.getenv("CUDAVERSE_CANDIDATE_MANIFEST", unset = "")
if (!nzchar(input) || !file.exists(input)) {
  stop("Set CUDAVERSE_CANDIDATE_INPUT to an existing input JSON.",
       call. = FALSE)
}
if (!nzchar(output)) {
  stop("Set CUDAVERSE_CANDIDATE_MANIFEST to an output path.",
       call. = FALSE)
}
if (file.exists(output)) {
  stop("Refusing to overwrite an existing candidate manifest.",
       call. = FALSE)
}
bundle_root <- normalizePath(dirname(output), winslash = "/", mustWork = TRUE)
input_root <- normalizePath(dirname(input), winslash = "/", mustWork = TRUE)
if (!identical(bundle_root, input_root)) {
  stop("Candidate input and manifest must share one evidence-bundle root.",
       call. = FALSE)
}

git_line <- function(args) {
  value <- system2("git", args, stdout = TRUE, stderr = TRUE)
  status <- attr(value, "status", exact = TRUE)
  if (!is.null(status) && status != 0L) {
    stop("git command failed: ", paste(value, collapse = "\n"),
         call. = FALSE)
  }
  unname(value)
}
commit <- git_line(c("rev-parse", "HEAD"))[[1L]]
branch <- git_line(c("branch", "--show-current"))[[1L]]
changes <- git_line(c("status", "--porcelain", "--untracked-files=all"))
version <- unname(read.dcf("DESCRIPTION", fields = "Version")[[1L]])
if (!identical(branch, "develop/native-cuda")) {
  stop("Final candidate evidence must be built on develop/native-cuda.",
       call. = FALSE)
}
if (length(changes)) {
  stop("Final candidate evidence requires a clean source tree.",
       call. = FALSE)
}
if (!grepl("^0\\.3\\.0(\\.9000)?$", version)) {
  stop("Final candidate evidence requires a 0.3.0 candidate version.",
       call. = FALSE)
}

frozen_commit <- "59e15c8c5a56d26e09a594886c875b1b8249f6f9"
remote_ref <- function(name) {
  value <- git_line(c("ls-remote", "origin", name))
  if (length(value) != 1L) stop("Unable to resolve frozen ref ", name,
                                call. = FALSE)
  strsplit(value[[1L]], "[[:space:]]+")[[1L]][[1L]]
}
main_ref <- remote_ref("refs/heads/main")
release_ref <- remote_ref("refs/heads/release/cran-0.1.0")
if (!identical(main_ref, frozen_commit) ||
    !identical(release_ref, frozen_commit)) {
  stop("A frozen release ref changed before candidate assembly.",
       call. = FALSE)
}

spec <- jsonlite::read_json(input, simplifyVector = FALSE)
spec$candidate <- list(
  source_commit = commit, version = version,
  branch = branch, clean = TRUE
)
spec$frozen_refs <- list(
  main = main_ref, release_cran_0_1_0 = release_ref
)
manifest <- build_candidate_evidence_manifest(spec, bundle_root)
staging <- tempfile(
  pattern = ".candidate-manifest-staging-",
  tmpdir = bundle_root, fileext = ".json"
)
on.exit(unlink(staging, force = TRUE), add = TRUE)
jsonlite::write_json(
  manifest, staging, auto_unbox = TRUE, pretty = TRUE, digits = 16,
  null = "null", na = "null"
)
old_manifest <- Sys.getenv("CUDAVERSE_CANDIDATE_MANIFEST", unset = NA_character_)
on.exit({
  Sys.unsetenv("CUDAVERSE_CANDIDATE_MANIFEST")
  if (!is.na(old_manifest)) {
    Sys.setenv(CUDAVERSE_CANDIDATE_MANIFEST = old_manifest)
  }
}, add = TRUE)
Sys.setenv(CUDAVERSE_CANDIDATE_MANIFEST = staging)
sys.source(
  file.path("tools", "check-candidate-evidence.R"),
  envir = new.env(parent = globalenv())
)
if (!file.rename(staging, output)) {
  stop("Validated candidate manifest could not be installed atomically.",
       call. = FALSE)
}
message("Built exact candidate evidence manifest: ", output)
