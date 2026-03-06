# Copilot Instructions for `aspire`

## Overview

`aspire` (Accelerated Sequential Posterior Inference via REuse) is a Python
framework for reusing existing posterior samples to obtain new Bayesian
inference results at a reduced computational cost. The PyPI package name is
`aspire-inference`, but the import name is `aspire`.

## Repository Layout

```
src/aspire/          # Main package source
  aspire.py          # Core Aspire class
  samples.py         # Samples / BaseSamples dataclasses
  transforms.py      # Data transforms (whitening, logit, etc.)
  history.py         # Training history tracking
  plot.py            # Plotting helpers
  utils.py           # Shared utilities (array API helpers, file I/O)
  flows/             # Normalizing flow wrappers
    base.py          # Abstract Flow base class
    torch/           # Zuko-backed flows (PyTorch)
    jax/             # FlowJax-backed flows (JAX)
  samplers/          # Posterior samplers
    base.py          # Abstract Sampler base class
    importance.py    # Importance sampler
    mcmc.py          # MCMC sampler (emcee)
    smc/             # Sequential Monte Carlo sampler
tests/               # pytest test suite
  conftest.py        # Shared fixtures (xp, dtype, rng)
  integration_tests/ # End-to-end tests
  test_flows/        # Flow-specific tests
  test_*.py          # Unit tests
examples/            # Runnable example scripts
docs/                # ReadTheDocs documentation source
```

## Key Design Decisions

- **Array-API agnostic**: the code targets the Python
  [Array API standard](https://data-apis.org/array-api/latest/) via
  `array-api-compat`. Always prefer `xp.*` calls (where `xp` is the resolved
  namespace) over hard-coding `numpy.*` or `torch.*`.
- **Backends**: `numpy` (default/testing), `jax.numpy` (JAX), and
  `array_api_compat.torch` (PyTorch). The active namespace is stored on
  `Aspire` / `Samples` instances as `self.xp`.
- **Flow backends**: `zuko` (PyTorch, default) and `flowjax` (JAX). Custom
  backends can be registered via the `aspire.flows` entry-point group.
- **`Samples` / `BaseSamples`**: Central data containers (dataclasses).
  `log_likelihood`, `log_prior`, and `log_q` are optional array fields.
  Functions passed to `Aspire` receive a `Samples` object and must return an
  array.

## Development Setup

```bash
# Install all optional dependencies for local development and tests
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,torch,jax,minipcn,emcee,blackjax,pandas]"
```

## Linting

The project uses [ruff](https://docs.astral.sh/ruff/) with a line length of
79 characters.

```bash
ruff check .          # check for linting errors
ruff format .         # auto-format code
ruff format --check . # check formatting without modifying (used in CI)
```

Pre-commit hooks (trailing whitespace, EOF fixer, YAML check, ruff, codespell)
are configured in `.pre-commit-config.yaml`.

## Running Tests

```bash
python -m pytest                   # run the full test suite with coverage
python -m pytest tests/test_samples.py   # run a single test file
python -m pytest -k "test_name"    # run tests matching a pattern
```

Tests run against Python 3.10–3.13 on Ubuntu in CI
(`.github/workflows/tests.yml`). The `conftest.py` parametrises the `xp`
fixture over `["jax", "torch", "numpy"]` and `dtype` over
`["float32", "float64", None]`.

Note: the `tests.yml` CI workflow installs PyTorch from the CPU-only index
(`https://download.pytorch.org/whl/cpu`) to keep downloads small.

## CI Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `tests.yml` | push/PR to `main`, `release*` | pytest matrix (Python 3.10–3.13) |
| `lint.yml` | push/PR to `main`, `release*` | `ruff format --check` |
| `examples.yml` | push/PR to `main`, `release*` | runs example scripts |
| `publish.yml` | release events | publishes to PyPI |

## Common Patterns

### Adding a new sampler

1. Subclass `aspire.samplers.base.Sampler`.
2. Implement the `sample(n_samples)` method; it must return a `Samples` object.
3. Register it in `aspire/samplers/__init__.py`.
4. Add unit tests under `tests/`.

### Adding a new flow backend

1. Subclass `aspire.flows.base.Flow`.
2. Set a class-level `xp` attribute pointing to the array namespace module.
3. Register it via the `aspire.flows` entry-point group in `pyproject.toml`.

### Working with the Array API

- Use `array_api_compat.array_namespace(x)` to get the namespace from an
  existing array.
- Helper utilities live in `aspire/utils.py` (`asarray`, `convert_dtype`,
  `resolve_dtype`, `to_numpy`, etc.).

## Known Issues / Workarounds

- **JAX float64**: JAX defaults to 32-bit floats. Tests enable 64-bit via
  `jax.config.update("jax_enable_x64", True)` in `conftest.py`. Do the same
  in scripts that require float64.
- **SciPy Array API**: `os.environ["SCIPY_ARRAY_API"] = "1"` must be set
  before importing scipy when using non-NumPy backends. `conftest.py` handles
  this automatically; `aspire/__init__.py` calls `enable_scipy_array_api()` on
  import.
- **PyTorch default dtype**: When using the `torch` backend, the default dtype
  is `torch.float32`. Pass `dtype="float64"` explicitly if higher precision is
  needed.
