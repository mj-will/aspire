import sys

import array_api_compat.numpy as xp
import numpy as np

from aspire.samplers.mcmc import Emcee, MiniPCN
from aspire.samples import MCMCSamples
from aspire.utils import AspireFile


class DummyPriorFlow:
    def sample_and_log_prob(self, n_samples: int):
        x = np.linspace(-1.0, 1.0, n_samples * 2).reshape(n_samples, 2)
        log_q = np.zeros(n_samples)
        return x, log_q

    def log_prob(self, x):
        x = np.asarray(x)
        return np.zeros(len(x))


def _log_likelihood(samples):
    return -np.sum(np.asarray(samples.x) ** 2, axis=1)


def _log_prior(samples):
    return -0.5 * np.sum(np.asarray(samples.x) ** 2, axis=1)


def test_emcee_mcmc_saves_chain_checkpoint(monkeypatch, tmp_path):
    class FakeEnsembleSampler:
        def __init__(
            self, nwalkers, dims, log_prob_fn, vectorize=True, **kwargs
        ):
            _ = (log_prob_fn, vectorize, kwargs)
            self.nwalkers = nwalkers
            self.dims = dims
            self._chain = None

        def run_mcmc(self, z0, nsteps, **kwargs):
            _ = kwargs
            z0 = np.asarray(z0)
            self._chain = np.repeat(z0[None, :, :], nsteps, axis=0)

        def get_chain(self, discard=0):
            return self._chain[discard:]

    monkeypatch.setitem(
        sys.modules,
        "emcee",
        type("FakeEmceeModule", (), {"EnsembleSampler": FakeEnsembleSampler}),
    )

    sampler = Emcee(
        log_likelihood=_log_likelihood,
        log_prior=_log_prior,
        dims=2,
        prior_flow=DummyPriorFlow(),
        xp=xp,
    )
    checkpoint_file = tmp_path / "emcee_ckpt.h5"
    out = sampler.sample(
        n_samples=6,
        nwalkers=6,
        nsteps=4,
        checkpoint_file_path=str(checkpoint_file),
        checkpoint_every=1,
    )

    assert len(out) > 0
    assert sampler.last_checkpoint_state is not None
    assert sampler.last_checkpoint_state["iteration"] == 4
    assert sampler.last_checkpoint_state["meta"]["stage"] == "mcmc_chain"
    assert "samples" in sampler.last_checkpoint_state
    with AspireFile(checkpoint_file, "r") as fp:
        loaded = MCMCSamples.load(fp, path="checkpoint/mcmc_chain")
        assert fp["checkpoint/mcmc_chain"].attrs["iteration"] == 4
        assert fp["checkpoint/mcmc_chain"].attrs["stage"] == "mcmc_chain"
        assert tuple(
            fp["checkpoint/mcmc_chain"].attrs["chain_shape"]
        ) == tuple(out.chain_shape)
    assert len(loaded) == len(out)


def test_minipcn_mcmc_saves_chain_checkpoint(monkeypatch, tmp_path):
    class FakeMiniPCNSampler:
        def __init__(
            self,
            log_prob_fn,
            step_fn,
            rng,
            dims,
            target_acceptance_rate,
        ):
            _ = (log_prob_fn, step_fn, rng, dims, target_acceptance_rate)

        def sample(self, z0, n_steps=100):
            z0 = np.asarray(z0)
            chain = np.repeat(z0[None, :, :], n_steps, axis=0)
            return chain, {}

    monkeypatch.setitem(
        sys.modules,
        "minipcn",
        type("FakeMiniPCNModule", (), {"Sampler": FakeMiniPCNSampler}),
    )

    sampler = MiniPCN(
        log_likelihood=_log_likelihood,
        log_prior=_log_prior,
        dims=2,
        prior_flow=DummyPriorFlow(),
        xp=xp,
    )
    checkpoint_file = tmp_path / "minipcn_ckpt.h5"
    out = sampler.sample(
        n_samples=6,
        n_steps=5,
        burnin=1,
        thin=2,
        checkpoint_file_path=str(checkpoint_file),
        checkpoint_every=1,
    )

    assert len(out) > 0
    assert sampler.last_checkpoint_state is not None
    assert sampler.last_checkpoint_state["iteration"] == 5
    assert sampler.last_checkpoint_state["meta"]["stage"] == "mcmc_chain"
    assert "samples" in sampler.last_checkpoint_state
    with AspireFile(checkpoint_file, "r") as fp:
        loaded = MCMCSamples.load(fp, path="checkpoint/mcmc_chain")
        assert fp["checkpoint/mcmc_chain"].attrs["iteration"] == 5
        assert fp["checkpoint/mcmc_chain"].attrs["stage"] == "mcmc_chain"
    assert len(loaded) > len(out)
