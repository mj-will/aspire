"""Example using sequential posterior inference with SMC.

This examples is slightly contrived, using a mixture of two Gaussians in 4D
as the target distribution. The goal is to demonstrate the ability of SMC to
explore multi-modal distributions, even when the initial samples deviate
significantly from the true modes.

In practice, one would ideally use more informative initial samples.
"""

from pathlib import Path

import numpy as np

from aspire import Aspire
from aspire.plot import plot_comparison
from aspire.samples import Samples
from aspire.utils import configure_logger

# RNG for generating initial samples
rng = np.random.default_rng(42)

# Output directory
outdir = Path("outdir") / "smc_example"
outdir.mkdir(parents=True, exist_ok=True)

# Configure logger to show INFO level messages
configure_logger()

# Number of dimensions
dims = 4

# Means and covariances of the two Gaussian components
mu1 = 2 * np.ones(dims)
mu2 = -2 * np.ones(dims)
cov1 = 0.5 * np.eye(dims)
cov2 = np.eye(dims)


def log_likelihood(samples):
    """Log-likelihood of a mixture of two Gaussians"""
    x = samples.x
    comp1 = (
        -0.5 * ((x - mu1) @ np.linalg.inv(cov1) * (x - mu1)).sum(axis=-1)
        - 0.5 * dims * np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(cov1)[1]
    )
    comp2 = (
        -0.5 * ((x - mu2) @ np.linalg.inv(cov2) * (x - mu2)).sum(axis=-1)
        - 0.5 * dims * np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(cov2)[1]
    )
    return np.logaddexp(comp1, comp2)  # Log-sum-exp for numerical stability


def log_prior(samples):
    """Standard normal prior"""
    return -0.5 * (samples.x**2).sum(axis=-1) - dims * 0.5 * np.log(2 * np.pi)


# Generate prior samples for comparison, these are not used in SMC
prior_samples = Samples(rng.normal(0, 1, size=(5000, dims)))

# We draw initial samples from two Gaussians centered away from the true modes
# to demonstrate the ability of SMC to explore the posterior
offset_1 = rng.uniform(-3, 3, size=(dims,))
offset_2 = rng.uniform(-3, 3, size=(dims,))
initial_samples = np.concatenate(
    [
        rng.normal(mu1 - offset_1, 1, size=(2500, dims)),
        rng.normal(mu2 - offset_2, 1, size=(2500, dims)),
    ],
    axis=0,
)
initial_samples = Samples(initial_samples)

# Initialize Aspire with the log-likelihood and log-prior
aspire = Aspire(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=dims,
    flow_class="NSF",  # Use Neural Spline Flow from zuko (default backend)
)

# Fit the normalizing flow to the initial samples
fit_history = aspire.fit(initial_samples, n_epochs=30)
# Plot loss
fit_history.plot_loss().savefig(outdir / "loss.png")

# Sample from the posterior using SMC
samples, history = aspire.sample_posterior(
    sampler="smc",  # Sequential Monte Carlo, this uses the default minipcn sampler
    n_samples=500,  # Number of particles in SMC
    n_final_samples=5000,  # Number of samples to draw from the final distribution
    sampler_kwargs=dict(  # Keyword arguments for the specific sampler
        n_steps=20,  # MCMC steps per SMC iteration
    ),
    return_history=True,  # To return the SMC history (e.g., ESS, betas)
)
# Plot SMC diagnostics
history.plot().savefig(outdir / "smc_diagnostics.png")

# Plot corner plot of the samples
# Include initial samples and prior samples for comparison
plot_comparison(
    initial_samples,
    prior_samples,
    samples,
    labels=["Initial Samples", "Prior Samples", "SMC Samples"],
).savefig(outdir / "posterior.png")
