# Contributing to Aspire

Contributions to Aspire are welcome, including bug reports, tests, documentation,
performance improvements, and inference functionality. This guide describes the
development workflow and conventions for contributions.

## Community and responsibility

Treat other contributors with respect and follow the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).
Contributions must comply with Aspire's [license](https://github.com/mj-will/aspire/blob/main/LICENSE).
Preserve required attribution when incorporating third-party material.

AI-assisted contributions are welcome under our
[AI policy](https://github.com/mj-will/aspire/blob/main/AI_POLICY.md).
Contributors are responsible for understanding, validating, and explaining their
changes, including numerical correctness and performance.

## Scope of contributions

Keep each pull request focused on one problem or a closely related set of changes.
Discuss substantial API changes, new dependencies, or new algorithms in an issue
before investing significant implementation effort. Small fixes can go directly
to a pull request.

Aspire helps reuse existing posterior samples to accelerate Bayesian inference.
Keep changes focused on this workflow and use existing sampling and density
interfaces where possible. Avoid unrelated refactors or duplicating functionality
already supplied by dependencies.

## Development setup

Aspire requires Python 3.11 or later. Use Pixi for development; follow the
[Pixi installation instructions](https://pixi.prefix.dev/latest/installation/)
to install it first. Clone the repository and create a contribution branch:

```bash
git clone https://github.com/mj-will/aspire.git
cd aspire
git switch -c fix-descriptive-name
pixi install
pixi run pre-commit install
```

If you do not have write access, fork the repository, clone your fork instead,
and add `https://github.com/mj-will/aspire.git` as an `upstream` remote. Keep
your local `main` up to date with the main Aspire repository and create a new
branch for each contribution. Do not develop directly on `main`.

Dependencies, environments, and tasks are defined in `pyproject.toml`. The
default environment uses Python 3.13 and installs Aspire in editable mode, its
test dependencies, CPU PyTorch and JAX flow backends, and lint tools. Source
changes take effect without reinstalling the package. The import namespace is
`aspire`; the distribution name is `aspire-inference`.

Pixi resolves dependencies and generates `pixi.lock` locally. When changing
dependencies, edit `pyproject.toml` and run `pixi install` again. Do not commit
generated lock files, `.pixi/`, caches, or test outputs. The installed pre-commit
hook runs code and file checks; it does not run pytest or validate commit messages.

## Code style and typing

- Follow the surrounding module's conventions. Ruff handles formatting and import
  ordering; Python code uses four-space indentation and a 79-character line limit.
- Use descriptive names. Conventional mathematical notation is appropriate when
  its meaning is clear locally.
- Add parameter and return annotations to new functions and methods, including
  private helpers and tests where practical. Update callers when changing an
  interface. Use Python 3.11-compatible syntax such as `list[str]` and `X | None`.
- Reuse the existing array annotations and namespace helpers in the affected
  module. Preserve array backends, devices, dtypes, and batch shapes; avoid
  implicit NumPy conversions and global random state.
- Give callbacks explicit signatures and limit `Any` to genuinely dynamic
  interfaces. Use `TYPE_CHECKING` for annotation-only imports where needed to
  avoid introducing optional dependencies at import time.
- Document public functions and classes with NumPy-style docstrings, including
  array shapes, supported inputs, return values, and numerical assumptions.
- Explain non-obvious numerical choices in comments. Avoid comments that simply
  restate the implementation.
- Avoid repeated target evaluations, unnecessary copies, and unused histories.
  Support performance claims with reproducible measurements.

Standalone style checks are available as `pixi run lint` and
`pixi run format-check`. Aspire does not currently configure a mypy task or
type-checking hook; annotations still need review and runtime validation.

## Checks and tests

For code changes, run the configured checks before requesting review:

```bash
pixi run pre-commit run --all-files
pixi run test
```

Pre-commit covers Ruff linting and formatting, spelling, whitespace, YAML
validation, and large-file checks. It may modify files. Review those changes and
rerun the hooks before committing; do not bypass a failing hook.

Tests use pytest and live in `tests/`, with integration tests in
`tests/integration_tests/`. During development, run the relevant tests first:

```bash
pixi run test-unit
pixi run python -m pytest tests/test_samples.py
pixi run python -m pytest tests/integration_tests/test_checkpointing.py
```

Add regression tests for bug fixes and tests for new behavior. Test observable
outcomes and numerical invariants. Use fixed RNG seeds and justified numerical
tolerances for stochastic tests. Cover relevant edge cases such as zero support,
degenerate weights, particle shapes, dtype precision, and bounded or periodic
coordinates. Changes affecting target distributions, weights, evidence, RNG
behavior, or checkpoint formats require an explanation and appropriate tests.
Sampling changes should be checked against a known target; persistence changes
should exercise save/load and resume behavior.

CI runs the complete suite on Linux with Python 3.11, 3.12, and 3.13. Use the
corresponding local environments when checking version compatibility:

```bash
pixi run -e py311 test
pixi run -e py312 test
pixi run -e py313 test
```

The environments support Linux and Apple Silicon and include CPU backends.
Report the commands actually run and their results, including failures, skipped
backends, and checks you could not complete. A skipped test does not verify a
backend. Do not claim GPU coverage from these CPU environments.

## Building the documentation

Documentation lives in `docs/` and uses Sphinx. Build it from the repository root:

```bash
pixi run -e docs docs
```

Open `docs/_build/html/index.html` to inspect the result. Rebuild after changing
documentation or public API docstrings. For documentation-only changes, run
pre-commit and build the documentation; run affected code examples if you change
them. The contribution guide and AI policy are included from these root Markdown
files so GitHub and Sphinx share the same source.

## Commit messages and pull requests

Use uppercase SciPy-style prefixes for commit subjects and pull request titles:

```text
PREFIX(scope): short imperative summary
PREFIX: short imperative summary
```

Use `BUG` for correctness fixes, `ENH` for functionality, `PERF` for performance,
`API` for interface changes, `DEP` for deprecations, `MAINT` for maintenance,
`TST` for tests, `DOC` for documentation, `BLD` for packaging, `CI` for automation,
`REL` for releases, and `STY` for formatting. An optional scope names the affected
area, for example `smc` or `checkpoint`.

Start the summary with a lowercase imperative verb, omit its trailing period,
and aim for 72 characters or fewer. Examples:

```text
BUG(checkpoint): preserve sampler configuration on resume
DOC: document development setup
```

Make commits in coherent units. Push your branch and open a pull request against
`mj-will/aspire:main`:

```bash
git push --set-upstream origin fix-descriptive-name
```

Use a draft PR for work in progress. Explain the problem and resulting behavior,
link related issues, describe numerical and compatibility changes, and summarize
validation. Add a changelog entry under `Unreleased` for user-visible changes;
routine internal refactoring and CI maintenance do not normally need an entry.
Keep the title and description aligned with the final implementation.

Contributors should respond to review feedback themselves. Reviewers should
distinguish required changes from suggestions and raise unrelated findings
separately. Maintainers should wait for applicable CI and review discussions
before merging.

## Releases

Maintain [CHANGELOG.md](https://github.com/mj-will/aspire/blob/main/CHANGELOG.md)
using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Describe user-visible
changes under `Unreleased`, grouped as `Added`, `Changed`, `Deprecated`, `Removed`,
`Fixed`, or `Security`. Explicitly describe effects on scientific results, RNG
behavior, public APIs, and checkpoint compatibility.

Versions are derived from Git tags by `setuptools-scm`; do not hard-code a version
number. Maintainers prepare the release changelog and verify the intended commit,
CI results, and built distribution before tagging. Publishing a GitHub release
triggers `.github/workflows/publish.yml` to publish to PyPI. Verify that the
workflow completes successfully; published distributions cannot be overwritten.
