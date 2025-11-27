"""Example using sequential posterior inference with SMC.

This example uses JAX for computations and BlackJAX for the MCMC sampling
in SMC step.
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from aspire import Aspire
from aspire.plot import plot_comparison
from aspire.samples import Samples
from aspire.utils import configure_logger

# RNG for generating initial samples
key = jax.random.key(42)

# Output directory
outdir = Path("outdir") / "blackjax_smc_example"
outdir.mkdir(parents=True, exist_ok=True)

# Configure logger to show INFO level messages
configure_logger()

# Number of dimensions
dims = 4
# Parameter names
parameters = [f"x{i}" for i in range(dims)]
# Prior bounds
prior_bounds = {param: (-5, 5) for param in parameters}

# Means and covariances of the two Gaussian components
mu1 = 2 * jnp.ones(dims)
mu2 = -2 * jnp.ones(dims)
cov1 = 0.5 * jnp.eye(dims)
cov2 = jnp.eye(dims)


def log_likelihood(samples):
    """Log-likelihood of a mixture of two Gaussians"""
    x = samples.x
    comp1 = (
        -0.5 * ((x - mu1) @ jnp.linalg.inv(cov1) * (x - mu1)).sum(axis=-1)
        - 0.5 * dims * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.linalg.slogdet(cov1)[1]
    )
    comp2 = (
        -0.5 * ((x - mu2) @ jnp.linalg.inv(cov2) * (x - mu2)).sum(axis=-1)
        - 0.5 * dims * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.linalg.slogdet(cov2)[1]
    )
    return jnp.logaddexp(comp1, comp2)  # Log-sum-exp for numerical stability


def log_prior(samples):
    """Uniform prior between -5 and 5 in each dimension"""
    x = samples.x
    in_bounds = jnp.all((x >= -5) & (x <= 5), axis=-1)
    logp = jnp.where(in_bounds, -dims * jnp.log(10), -jnp.inf)
    return logp


# Generate prior samples for comparison, these are not used in SMC
key, prior_key = jax.random.split(key)
prior_samples = Samples(
    jax.random.uniform(prior_key, shape=(5000, dims), minval=-5, maxval=5),
    parameters=parameters,
)

# True posterior samples for comparison
key, post_key0, post_key1 = jax.random.split(key, 3)
true_posterior_samples = Samples(
    jnp.concatenate(
        [
            jax.random.multivariate_normal(
                post_key0, mu1, cov1, shape=(2500,)
            ),
            jax.random.multivariate_normal(
                post_key1, mu2, cov2, shape=(2500,)
            ),
        ],
        axis=0,
    ),
    parameters=parameters,
)

# We draw initial samples from two Gaussians centered away from the true modes
# to demonstrate the ability of SMC to explore the posterior
key, offset_key1, offset_key2 = jax.random.split(key, 3)
offset_1 = jax.random.uniform(offset_key1, shape=(dims,), minval=-3, maxval=3)
offset_2 = jax.random.uniform(offset_key2, shape=(dims,), minval=-3, maxval=3)
key, init_key1, init_key2 = jax.random.split(key, 3)
initial_samples = jnp.concatenate(
    [
        jax.random.normal(init_key1, shape=(2500, dims)) + (mu1 - offset_1),
        jax.random.normal(init_key2, shape=(2500, dims)) + (mu2 - offset_2),
    ],
    axis=0,
)
initial_samples = Samples(initial_samples, parameters=parameters)

# Initialize Aspire with the log-likelihood and log-prior
key, aspire_key = jax.random.split(key)
aspire = Aspire(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=dims,
    flow_backend="flowjax",  # Use Flowjax as the normalizing flow backend
    prior_bounds=prior_bounds,  # Specify prior bounds
    parameters=parameters,
    key=aspire_key,
)

# Fit the normalizing flow to the initial samples
fit_history = aspire.fit(initial_samples, max_epochs=30)

# Plot loss
fit_history.plot_loss().savefig(outdir / "loss.png")

# Sample from the posterior using SMC
# We use BlackJAX's NUTS as the MCMC kernel within SMC
# We enable the bounded to unbounded transform in the preconditioning to avoid
# issues with NUTS on bounded spaces
samples, history = aspire.sample_posterior(
    sampler="blackjax_smc",  # use the BlackJAX SMC sampler
    n_samples=500,  # Number of particles in SMC
    n_final_samples=5000,  # Number of samples to draw from the final distribution
    adaptive=True,
    target_efficiency=0.8,
    sampler_kwargs=dict(  # Keyword arguments for the specific sampler
        algorithm="nuts",  # Use NUTS within SMC
        step_size=0.1,  # Step size for NUTS, this will need tuning
        n_steps=20,  # Number of leapfrog steps for NUTS
    ),
    preconditioning_kwargs=dict(
        affine_transform=True,  # Use affine transform preconditioning
        bounded_to_unbounded=True,  # Transform bounded parameters to unbounded space
    ),
    return_history=True,  # To return the SMC history (e.g., ESS, betas)
)
# Plot SMC diagnostics
history.plot().savefig(outdir / "smc_diagnostics.png")

# Plot corner plot of the samples
# Include initial samples and prior samples for comparison
plot_comparison(
    initial_samples,
    true_posterior_samples,
    samples,
    labels=["Initial Samples", "True Posterior Samples", "SMC Samples"],
    per_samples_kwargs=[
        {"color": "grey"},
        {"color": "k"},
        {"color": "C1"},
    ],
).savefig(outdir / "posterior.png")
