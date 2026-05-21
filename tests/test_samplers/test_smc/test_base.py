from unittest.mock import MagicMock

import array_api_compat.numpy as np
import pytest

from aspire.samplers.smc.base import SMCSampler


@pytest.fixture
def dims():
    return 2


@pytest.fixture
def sampler(dims, rng):
    sampler = SMCSampler(
        log_prior=MagicMock(return_value=np.zeros(10)),
        log_likelihood=MagicMock(return_value=np.zeros(10)),
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


def test_determine_beta(sampler):
    # Test that beta is determined correctly based on the log likelihoods
    log_likelihoods = np.array([-10, -5, 0])
    sampler.log_likelihood = MagicMock(return_value=log_likelihoods)
    sampler.determine_beta()
    assert sampler.beta == 1.0
