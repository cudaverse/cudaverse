workflow <- file.path(".github", "workflows", "pkgdown.yaml")
if (!file.exists(workflow)) {
  stop("The pkgdown workflow does not exist.", call. = FALSE)
}

lines <- readLines(workflow, warn = FALSE)
deploy_start <- grep("^  deploy:[[:space:]]*$", lines)
if (length(deploy_start) != 1L) {
  stop("Expected exactly one pkgdown deploy job.", call. = FALSE)
}

later_jobs <- grep("^  [[:alnum:]_-]+:[[:space:]]*$", lines)
later_jobs <- later_jobs[later_jobs > deploy_start]
deploy_end <- if (length(later_jobs)) later_jobs[[1L]] - 1L else length(lines)
deploy_block <- paste(lines[deploy_start:deploy_end], collapse = "\n")

required <- c(
  "github.event_name == 'push'",
  "github.ref == 'refs/heads/main'",
  "github.event_name == 'release'",
  "Deploy to GitHub Pages"
)
missing <- required[!vapply(
  required, grepl, logical(1L), x = deploy_block, fixed = TRUE
)]
if (length(missing)) {
  stop(
    "The pkgdown deploy job is missing release boundary: ",
    paste(missing, collapse = ", "), call. = FALSE
  )
}
if (grepl("github.event_name != 'pull_request'", deploy_block, fixed = TRUE)) {
  stop(
    "The pkgdown deploy job must not deploy arbitrary manual workflows.",
    call. = FALSE
  )
}

message(
  "External-release workflow boundaries passed: manual pkgdown runs build ",
  "evidence without deploying Pages."
)
