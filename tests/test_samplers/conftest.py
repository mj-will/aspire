import numpy as np
import pytest
from array_api_compat import numpy as xp


class GaussianFlow:
    xp = xp

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def sample_and_log_prob(self, n_samples, context=None, *, rng=None):
        del context
        generator = self.rng if rng is None else rng
        particles = xp.asarray(generator.normal(size=(n_samples, 2)))
        return particles, self.log_prob(particles)

    def sample(self, n_samples, context=None, *, rng=None):
        return self.sample_and_log_prob(
            n_samples,
            context=context,
            rng=rng,
        )[0]

    def log_prob(self, particles, context=None):
        del context
        return -0.5 * xp.sum(particles**2, axis=-1)


def log_prior(samples):
    return xp.where(
        xp.all(xp.abs(samples.x) < 10, axis=-1),
        -0.5 * xp.sum(samples.x**2, axis=-1),
        -xp.inf,
    )


def log_likelihood(samples):
    return -0.5 * xp.sum((samples.x - 1.0) ** 2, axis=-1)


@pytest.fixture
def sampler_kwargs():
    return {
        "log_likelihood": log_likelihood,
        "log_prior": log_prior,
        "dims": 2,
        "proposal": GaussianFlow(),
        "xp": xp,
        "parameters": ["x", "y"],
    }
