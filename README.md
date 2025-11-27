# aspire: Accelerated Sequential Posterior Inference via REuse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15658747.svg)](https://doi.org/10.5281/zenodo.15658747)
[![PyPI](https://img.shields.io/pypi/v/aspire-inference)](https://pypi.org/project/aspire-inference/)
[![Documentation Status](https://readthedocs.org/projects/aspire/badge/?version=latest)](https://aspire.readthedocs.io/en/latest/?badge=latest)
![tests](https://github.com/mj-will/aspire/actions/workflows/tests.yml/badge.svg)


aspire is a framework for reusing existing posterior samples to obtain new results at a reduced cost.

## Installation

aspire can be installed from PyPI using `pip`

```
pip install aspire-inference
```

**Important:** the name of `aspire` on PyPI is `aspire-inference` but once installed
the package can be imported and used as `aspire`.

## Quickstart

```python
import numpy as np
from aspire import Aspire, Samples

# Define a log-likelihood and log-prior
def log_likelihood(samples):
    x = samples.x
    return -0.5 * np.sum(x**2, axis=-1)

def log_prior(samples):
    return -0.5 * np.sum(samples.x**2, axis=-1)

# Create the initial samples
init = Samples(np.random.normal(size=(2_000, 4)))

# Define the aspire object
aspire = Aspire(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=4,
    parameters=[f"x{i}" for i in range(4)],
)

# Fit the normalizing flow
aspire.fit(init, n_epochs=20)

# Sample the posterior
posterior = aspire.sample_posterior(
    sampler="smc",
    n_samples=500,
    sampler_kwargs=dict(n_steps=100),
)

# Plot the posterior distribution
posterior.plot_corner()
```

## Documentation

See the [documentation on ReadTheDocs][docs].

## Citation

If you use `aspire` in your work please cite the [DOI][DOI] and [paper][paper].


[docs]: https://aspire.readthedocs.io/
[DOI]: https://doi.org/10.5281/zenodo.15658747
[paper]: https://arxiv.org/abs/2511.04218
