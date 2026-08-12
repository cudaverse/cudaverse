candidate_release_policies <- function() {
  list(
    list(
      line = "0.3",
      branch = "develop/native-cuda",
      version_pattern = "^0\\.3\\.0(\\.9000)?$"
    ),
    list(
      line = "0.4",
      branch = "develop/0.4",
      version_pattern = "^0\\.4\\.0(\\.9000)?$"
    )
  )
}

candidate_release_policy <- function(version, branch) {
  matches <- Filter(
    function(policy) {
      identical(branch, policy$branch) &&
        grepl(policy$version_pattern, version)
    },
    candidate_release_policies()
  )
  if (length(matches) == 1L) matches[[1L]] else NULL
}

candidate_release_policy_description <- function() {
  paste(
    vapply(
      candidate_release_policies(),
      function(policy) paste0(policy$branch, " (", policy$line, ".0)"),
      character(1L)
    ),
    collapse = ", "
  )
}
