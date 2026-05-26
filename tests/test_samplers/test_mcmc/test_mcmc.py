from unittest.mock import MagicMock

import numpy as np
import pytest

from aspire.samplers.mcmc import MCMCSampler


@pytest.fixture
def dims():
    return 2


def log_likelihood(samples):
    return -0.5 * np.sum(samples.x**2, axis=1)


def log_prior(samples):
    return -0.5 * np.sum(samples.x**2, axis=1)


@pytest.fixture
def sampler(dims, rng):
    sampler = MCMCSampler(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        dims=dims,
        preconditioning_transform=None,
        prior_flow=MagicMock(
            sample_and_log_prob=MagicMock(
                return_value=(rng.random((10, dims)), rng.random(10))
            ),
            xp=np,
        ),
        xp=np,
    )
    return sampler


def test_draw_initial_samples(sampler):
    n_samples = 10
    samples = sampler.draw_initial_samples(n_samples)
    assert samples.x.shape[0] == n_samples
    assert samples.parameters is not None


def test_draw_initial_samples_invalid_flow(sampler, rng):
    n_samples = 10
    samples = rng.random((n_samples, sampler.dims))
    log_prob = rng.random(n_samples)
    log_prob[0] = -np.inf  # Simulate flow sampling failure
    sampler.prior_flow.sample_and_log_prob = MagicMock(
        return_value=(samples, log_prob)
    )
    with pytest.raises(
        ValueError, match="Proposal returned non-finite log probabilities."
    ):
        sampler.draw_initial_samples(n_samples)


def test_draw_initial_samples_invalid_log_prior(sampler, rng):
    def log_prior(samples):
        # Random subset of samples have zero prior probability
        return np.where(
            rng.choice([True, False], size=samples.x.shape[0]), -np.inf, 0.0
        )

    sampler.log_prior = log_prior
    n_samples = 10
    sampler.draw_initial_samples(n_samples)


def test_draw_initial_samples_invalid_log_likelihood(sampler, rng):
    def log_likelihood(samples):
        # Random subset of samples have zero likelihood
        return np.where(
            rng.choice([True, False], size=samples.x.shape[0]), -np.inf, 0.0
        )

    sampler.log_likelihood = log_likelihood
    n_samples = 10
    sampler.draw_initial_samples(n_samples)
