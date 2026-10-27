import pytest
from sequince.checkpoint import SMCCheckpoint

from aspire.samplers.mcmc import Emcee, MCMCSampler
from aspire.samplers.smc import EmceeSMC
from aspire.samples import MCMCSamples


def test_mcmc_returns_aspire_samples_and_history(sampler_kwargs):
    sampler = MCMCSampler(**sampler_kwargs)
    samples = sampler.sample(8, n_steps=3, rng=5)

    assert isinstance(samples, MCMCSamples)
    assert samples.x.shape == (8, 2)
    assert sampler.result.chain.shape == (3, 8, 2)
    assert sampler.history is sampler.result


def test_mcmc_supports_native_and_checkpoint_callbacks(sampler_kwargs):
    iterations = []
    checkpoints = []

    sampler = MCMCSampler(**sampler_kwargs)
    sampler.sample(
        8,
        n_steps=3,
        rng=5,
        callbacks=[lambda state, context, result: iterations.append(result)],
        checkpoint_callback=checkpoints.append,
    )

    assert len(iterations) == 1
    assert len(checkpoints) == 1
    assert isinstance(checkpoints[0], SMCCheckpoint)


@pytest.mark.requires("emcee")
@pytest.mark.parametrize("vectorize", [False, True])
def test_emcee_mcmc_and_smc(sampler_kwargs, vectorize: bool):
    mcmc = Emcee(**sampler_kwargs)
    mcmc_samples = mcmc.sample(8, n_steps=2, rng=12, vectorize=vectorize)
    smc = EmceeSMC(**sampler_kwargs)
    smc_samples = smc.sample(
        8,
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"nsteps": 1, "vectorize": vectorize},
        rng=13,
    )

    assert mcmc_samples.x.shape == (8, 2)
    assert mcmc.result.chain.shape == (2, 8, 2)
    assert smc_samples.x.shape == (8, 2)
    assert smc_samples.beta == 1.0
    assert len(smc.history.mcmc_acceptance) == len(smc.result.states)
    assert smc.history.mcmc_autocorr == []
