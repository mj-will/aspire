# Agent instructions

Before making changes, read and follow [CONTRIBUTING.md](CONTRIBUTING.md) and
[AI_POLICY.md](AI_POLICY.md). CONTRIBUTING.md defines the development setup,
code style, typing, testing, commit-message, and pull request conventions.

## Working in the repository

- Inspect the working tree before editing. Preserve existing work and do not
  overwrite or revert unrelated changes. When committing, stage only files
  relevant to the requested task.
- Keep changes focused. Follow existing interfaces and helpers, and avoid
  unrelated refactors. Add dependencies or abstractions only when they address
  a concrete need within the task.
- Keep generated lock files, local environments, caches, and generated outputs
  out of commits unless explicitly requested. Preserve existing local files.

## Scientific correctness and validation

- Identify changes that affect target distributions, weighting, evidence
  estimates, RNG behavior, or checkpoint compatibility. Explain intentional
  behavior changes and support them with appropriate tests.
- Follow CONTRIBUTING.md for validation. Report the checks actually run and their
  results, including failures, skipped backends, and checks that could not be
  completed. Do not describe skipped checks as passing.
- Do not bypass hooks or disable checks to make a commit succeed. Address the
  failure, or report the blocker when it cannot be resolved within the task.
