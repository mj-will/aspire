# poppy

Posterior post-processing in python.

## Installation

Poppy can be installed from PyPI using `pip`

```
pip install poppy-inference
```

**Important:** the name of `poppy` on PyPI is `poppy-inference` but once installed
the package can be imported and used as `poppy`.


## Basic usage

`poppy` is designed around three key steps:

1. Defining a poppy instance
2. Fitting the normalizing flow given a set of initial samples
3. Sampling from the posterior distribution

The steps reflect how you use the code:

```python
from poppy import Poppy

# Define the log-likelihood and log-prior
...

poppy = Poppy(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=dims,                    # Number of dimensions
    parameters=parameters,        # Optional
    prior_bounds=prior_bounds,    # Optional but required for some other options
)

# Fit the normalizing flow to the initial samples
history = poppy.fit(initial_samples)

# Produce 5000 samples from the posterior
samples = poppy.sample_posterior(5000)
```

## Supported algorithms

`poppy` currently supports three algorithms for sampling from a posterior
distribution given an initial set of samples

- Importance sampling (default)
- Markov Chain Monte Carlo
- Informed Sequential Monte Carlo

The specific algorithm can be chosen by specifying the `sampler` keyword argument when calling `sample_posterior`.
See the sections below for details


### Importance Sampling

**Name:** `importance`

This method uses importance sampling with the trained flow as the proposal distribution and the posterior as the target.

The only configurable option when calling `sample_posterior` is the number of samples `n_samples`

**Note:** this method does not implement periodic boundary conditions or preconditioning with a normalizing flow.


### Markov Chain Monte Carlo

This method uses MCMC to sample from the posterior. By default, the trained flow is only used to draw the initial samples.

#### Samplers

- `emcee`:
- `minipcn`:

### Sequential Monte Carlo

This methods using informed sequential Monte Carlo to sample from the posterior with the trained flow serving
as the initial distribution.

#### Samplers

- `smc`: See `minipcn_smc`
- `minipcn_smc`:
- `emcee_smc`:
- `blackjax_smc`:

## Preconditioning

**Note:** this feature is largely untested

The trained normalizing flow can also be used as a preconditioning transforms when sampling with a method that uses an MCMC step (MCMC or SMC). The normalizing flow is used to reparameterize the space and improve sampling efficiency.

By default, basic preconditioning is applied, this includes options for transforming parameters from a bounded to an unbounded space, a periodic transform and an affine transform.

The normalizing flow preconditioning is enabled by setting `precondtioning=flow` when calling `sample_posterior` with a supported sampler. `preconditioning_kwargs` is used to configure the flow as you normally would when calling `Poppy`.
