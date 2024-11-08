import numpy as np
import torch
import torch.distributions

import poppy

poppy.set_backend("torch")
from poppy import Poppy
from poppy.samples.torch import TorchSamples
from poppy.utils import configure_logger

configure_logger("INFO")


dims = 16


def log_likelihood(samples):
    x = samples.x
    return torch.distributions.Normal(2, 1).log_prob(x).sum(axis=-1)


def log_prior(samples):
    x = samples.x
    return torch.distributions.Uniform(-10, 10).log_prob(x).sum(axis=-1)

true_log_evidence = -dims * np.log(20)

initial_samples = TorchSamples(1.0 * torch.randn(5000, dims) + 2.0)

parameters = [f"x_{i}" for i in range(dims)]
prior_bounds = {p: [-10, 10] for p in parameters}

# Define the poppy object
poppy = Poppy(
    log_likelihood=log_likelihood,
    log_prior=log_prior,
    dims=dims,
    parameters=parameters,
    prior_bounds=prior_bounds,
    flow_matching=True,
    hidden_features=4
    * [
        100,
    ],
)

# Fit the flow to the initial samples
print(initial_samples.x.min())
history = poppy.fit(
    initial_samples,
    n_epochs=500,
    lr_annealing=True,
)
print(initial_samples.x.min())
fig = history.plot_loss()
fig.savefig("loss.png")

# Produce samples from the posterior
samples = poppy.sample_posterior(2000).to_numpy()
print(f"True log evidence: {true_log_evidence}")

corner_kwargs = dict(
    density=True,
    bins=30,
    color="C0",
    hist_kwargs=dict(density=True, color="C0"),
)

fig = None
fig = poppy.training_samples.plot_corner(**corner_kwargs)
corner_kwargs["color"] = "lightgrey"
corner_kwargs["hist_kwargs"]["color"] = "lightgrey"
fig = samples.plot_corner(fig=fig, include_weights=False, **corner_kwargs)
corner_kwargs["color"] = "C1"
corner_kwargs["hist_kwargs"]["color"] = "C1"
fig = samples.plot_corner(fig=fig, **corner_kwargs)
fig.savefig("comparison.png")
