"""Check Aspire's adapters against a normalized conjugate Gaussian problem."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from array_api_compat import numpy as xp
from orng import RandomGenerator
from sequince.steps.context import StepContext

from aspire.samplers.mcmc import MCMCSampler
from aspire.samplers.smc import SMCSampler
from aspire.samples import Samples


class GaussianProposal:
    xp = xp

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator,
    ) -> tuple[np.ndarray, np.ndarray]:
        particles = rng.normal(size=(n_samples, 2))
        return particles, self.log_prob(particles)

    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RandomGenerator,
    ) -> np.ndarray:
        return self.sample_and_log_prob(n_samples, context, rng=rng)[0]

    def log_prob(
        self, particles: np.ndarray, context: StepContext | None = None
    ) -> np.ndarray:
        return -0.5 * np.sum(particles**2, axis=-1) - math.log(2 * math.pi)


def log_prior(samples: Samples) -> np.ndarray:
    return GaussianProposal().log_prob(samples.x)


def log_likelihood(samples: Samples) -> np.ndarray:
    return GaussianProposal().log_prob(samples.x - 1)


def sampler_kwargs() -> dict[str, Any]:
    return dict(
        proposal=GaussianProposal(),
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        dims=2,
        xp=xp,
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_smc_gaussian_moments_and_absolute_evidence(adaptive: bool) -> None:
    sampler = SMCSampler(**sampler_kwargs())
    samples = sampler.sample(
        4096,
        adaptive=adaptive,
        n_steps=None if adaptive else 4,
        sampler_kwargs={"n_steps": 5},
        rng=42,
    )
    # Two independent N(0,1) priors and N(x,1) observations equal to one.
    # Each posterior is N(1/2,1/2); the evidence is N(1;0,2)^2.
    assert samples.beta == 1.0
    np.testing.assert_allclose(np.mean(samples.x, axis=0), 0.5, atol=0.06)
    np.testing.assert_allclose(np.var(samples.x, axis=0), 0.5, atol=0.07)
    assert samples.log_evidence == pytest.approx(
        -math.log(4 * math.pi) - 0.5, abs=0.06
    )
    assert math.isfinite(samples.log_evidence_error)


def test_mcmc_gaussian_moments() -> None:
    sampler = MCMCSampler(**sampler_kwargs())
    samples = sampler.sample(n_walkers=64, n_steps=300, burnin=50, rng=42)
    np.testing.assert_allclose(np.mean(samples.x, axis=0), 0.5, atol=0.08)
    np.testing.assert_allclose(np.var(samples.x, axis=0), 0.5, atol=0.1)


@pytest.mark.parametrize("suffix", [".h5", ".pkl", ".pickle"])
def test_mcmc_checkpoint_extends_the_chain(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"mcmc{suffix}"
    first = MCMCSampler(**sampler_kwargs())
    first.sample(n_walkers=8, n_steps=3, rng=19, checkpoint_file_path=path)
    resumed = MCMCSampler(**sampler_kwargs())
    resumed.sample(n_walkers=8, n_steps=3, resume_from=path)
    assert resumed.result.chain.shape == (6, 8, 2)
    np.testing.assert_array_equal(resumed.result.chain[:3], first.result.chain)


@pytest.mark.parametrize("n_final_samples", [32, 64, 96])
@pytest.mark.parametrize("suffix", [".h5", ".pkl", ".pickle"])
def test_smc_checkpoint_replays_uninterrupted_run(
    tmp_path: Path, n_final_samples: int, suffix: str
) -> None:
    options = dict(
        max_beta_step=0.2,
        sampler_kwargs={"n_steps": 2},
        n_final_samples=n_final_samples,
    )
    full = SMCSampler(**sampler_kwargs())
    full_samples = full.sample(64, rng=71, **options)
    path = tmp_path / f"partial{suffix}"
    partial = SMCSampler(**sampler_kwargs())
    partial.sample(
        64, rng=71, max_n_steps=2, checkpoint_file_path=path, **options
    )
    resumed = SMCSampler(**sampler_kwargs())
    resumed_samples = resumed.sample(64, resume_from=path, **options)
    # Compare the committed SMC states, before final posterior resampling.
    np.testing.assert_array_equal(
        resumed.result.final_state.particles, full.result.final_state.particles
    )
    assert (
        resumed.result.final_state.log_evidence
        == full.result.final_state.log_evidence
    )
    assert resumed.history.beta == full.history.beta
    np.testing.assert_array_equal(
        resumed.history.log_norm_ratio, full.history.log_norm_ratio
    )
    np.testing.assert_array_equal(resumed_samples.x, full_samples.x)
