"""
This examples demonstrates how to use aspire to fit a flow to a simple Gaussian
likelihood with a uniform prior.
"""

import math
from pathlib import Path

from aspire import Aspire
from aspire.plot import plot_comparison
from aspire.samples import Samples
from aspire.utils import AspireFile, configure_logger
from scipy.stats import norm, uniform

# Configure the logger
configure_logger("INFO")

outdir = Path("outdir") / "basic_example"
outdir.mkdir(parents=True, exist_ok=True)

# Number of dimensions
dims = 4


# Define the log likelihood and log prior
def log_likelihood(samples: Samples):
    # The log likelihood must accept a Samples object
    # The samples object contains the samples in the attribute samples.x
    return norm(2, 1).logpdf(samples.x).sum(axis=-1)


def log_prior(samples: Samples):
    return uniform(-10, 20).logpdf(samples.x).sum(axis=-1)


# True evidence is analytic for a Gaussian likelihood and uniform prior
true_log_evidence = -dims * math.log(20)

# Generate some initial samples
# These are slightly biased compared to the true posterior
initial_samples = Samples(norm(2.5, 1.0).rvs(size=(5000, dims)))
# Define the parameters and prior bounds
parameters = [f"x_{i}" for i in range(dims)]
prior_bounds = {p: [-10, 10] for p in parameters}

# Define the aspire object
aspire = Aspire(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=dims,
    parameters=parameters,
    prior_bounds=prior_bounds,
)

# Fit the flow to the initial samples
history = aspire.fit(
    initial_samples,
    n_epochs=50,
)
# Plot the loss
fig = history.plot_loss()
fig.savefig(outdir / "loss.png")

# Produce samples from the posterior
samples = aspire.sample_posterior(5000)

# Save the the results to a file
# The AspireFile is a small wrapper around h5py.File that automatically
# includes additional metadata
with AspireFile(outdir / "aspire_result.h5", "w") as f:
    aspire.save_config(f, "aspire_config")
    samples.save(f, "posterior_samples")
    history.save(f, "flow_history")

fig = plot_comparison(
    initial_samples,
    samples,
    samples,
    per_samples_kwargs=[
        dict(include_weights=True, color="C0"),
        dict(include_weights=False, color="lightgrey"),
        dict(include_weights=True, color="C1"),
    ],
    labels=["Training samples", "Samples (w/o weights)", "Posterior samples"],
)

fig.savefig(outdir / "comparison.png")
