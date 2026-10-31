import numpy as np
import pytest
from sequince.proposals import FlowProposal

from aspire.samplers.importance import ImportanceSampler


def test_importance_sampler_reuses_cached_proposal(sampler_kwargs):
    sampler = ImportanceSampler(**sampler_kwargs)
    proposal = sampler._proposal()

    samples = sampler.sample(8, rng=3)

    assert sampler._proposal() is proposal
    assert isinstance(proposal, FlowProposal)
    assert samples.x.shape == (8, 2)
    assert samples.log_q.shape == (8,)


@pytest.mark.parametrize(
    ("particles", "log_prob", "message"),
    [
        (
            np.zeros((7, 2)),
            np.zeros(7),
            "exactly one particle per requested sample",
        ),
        (
            np.array([[np.inf, 0.0], *([[0.0, 0.0]] * 7)]),
            np.zeros(8),
            "non-finite particles",
        ),
        (
            np.zeros((8, 2)),
            np.zeros(7),
            "one value per particle",
        ),
        (
            np.zeros((8, 2)),
            np.zeros((8, 1)),
            "one value per particle",
        ),
        (
            np.zeros(8),
            np.zeros(8),
            r"must have shape \(n_samples, 2\)",
        ),
        (
            np.zeros((8, 3)),
            np.zeros(8),
            r"must have shape \(n_samples, 2\)",
        ),
        (
            np.zeros((8, 2)),
            np.array([-np.inf, *([0.0] * 7)]),
            "non-finite log densities",
        ),
        (
            np.zeros((8, 2)),
            np.array([np.nan, *([0.0] * 7)]),
            "non-finite log densities",
        ),
    ],
)
def test_importance_rejects_invalid_flow_outputs(
    sampler_kwargs,
    particles,
    log_prob,
    message,
):
    sampler = ImportanceSampler(**sampler_kwargs)
    sampler.prior_flow.sample_and_log_prob = lambda *args, **kwargs: (
        particles,
        log_prob,
    )

    with pytest.raises(ValueError, match=message):
        sampler.sample(8, rng=3)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("component", ["prior", "likelihood"])
def test_importance_rejects_invalid_target_outputs(
    sampler_kwargs,
    component,
    invalid,
):
    sampler = ImportanceSampler(**sampler_kwargs)

    def invalid_density(samples):
        values = np.zeros(len(samples))
        values[0] = invalid
        return values

    if component == "prior":
        sampler._log_prior = invalid_density
    else:
        sampler._log_likelihood = invalid_density

    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        sampler.sample(8, rng=3)


@pytest.mark.parametrize("component", ["prior", "likelihood"])
def test_importance_retains_zero_density_target_outputs(
    sampler_kwargs,
    component,
):
    sampler = ImportanceSampler(**sampler_kwargs)

    def zero_density(samples):
        values = np.zeros(len(samples))
        values[0] = -np.inf
        return values

    if component == "prior":
        sampler._log_prior = zero_density
    else:
        sampler._log_likelihood = zero_density

    samples = sampler.sample(8, rng=3)

    values = (
        samples.log_prior if component == "prior" else samples.log_likelihood
    )
    assert np.isneginf(values[0])
