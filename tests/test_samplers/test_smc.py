import warnings

import numpy as np
import pytest
from array_api_compat import numpy as xp
from sequince.checkpoint import SMCCheckpoint
from sequince.posterior import (
    draw_additional_samples as sequince_draw_additional_samples,
)

from aspire.aspire import Aspire
from aspire.history import SMCHistory
from aspire.samplers.smc import BlackJAXSMC, MiniPCNSMC, SMCSampler
from aspire.samples import SMCSamples


@pytest.mark.parametrize("via_aspire", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_beta_tolerance_is_accepted_with_deprecation_warning(
    sampler_kwargs, via_aspire: bool, adaptive: bool
) -> None:
    def sample(**extra_kwargs) -> SMCSamples:
        options = dict(
            adaptive=adaptive,
            n_steps=None if adaptive else 2,
            min_beta_step=0.05,
            max_beta_step=0.3,
            sampler_kwargs={"n_steps": 1},
            rng=19,
            **extra_kwargs,
        )
        if via_aspire:
            aspire = Aspire(
                log_prior=sampler_kwargs["log_prior"],
                log_likelihood=sampler_kwargs["log_likelihood"],
                dims=2,
                parameters=sampler_kwargs["parameters"],
                flow=type(sampler_kwargs["proposal"])(),
                xp=sampler_kwargs["xp"],
            )
            return aspire.sample_posterior(16, sampler="smc", **options)
        return SMCSampler(**sampler_kwargs).sample(16, **options)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        expected = sample()
    assert not any("beta_tolerance" in str(item.message) for item in caught)

    with pytest.warns(DeprecationWarning, match="beta_tolerance.*no effect"):
        actual = sample(beta_tolerance=1e-3)
    np.testing.assert_array_equal(actual.x, expected.x)
    assert actual.log_evidence == expected.log_evidence
    assert actual.beta == expected.beta == 1.0


def test_smc_returns_aspire_samples_and_history(sampler_kwargs):
    sampler = SMCSampler(**sampler_kwargs)
    samples = sampler.sample(
        16,
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"n_steps": 2},
        rng=4,
    )

    assert isinstance(samples, SMCSamples)
    assert samples.x.shape == (16, 2)
    assert samples.beta == 1.0
    assert samples.log_q.shape == (16,)
    assert samples.log_prior.shape == (16,)
    assert samples.log_likelihood.shape == (16,)
    assert samples.log_evidence == pytest.approx(
        sum(sampler.history.log_norm_ratio)
    )
    assert samples.log_evidence_error == pytest.approx(
        np.sqrt(sum(sampler.history.log_norm_ratio_var))
    )
    assert isinstance(sampler.history, SMCHistory)
    assert sampler.history is not sampler.result
    assert len(sampler.history.beta) == len(sampler.result.states) == 2
    assert len(sampler.history.sample_history) == 3
    assert sampler.history.sample_history[0].beta == 0.0
    assert sampler.result.initial_state.metadata["beta"] == 0.0
    assert sampler.history.sample_history[-1].beta == 1.0
    assert len(sampler.history.ess) == 2
    assert len(sampler.history.ess_target) == 2
    assert len(sampler.history.mcmc_acceptance) == 2
    assert np.all(np.isfinite(sampler.history.log_norm_ratio_var))
    assert np.all(np.isfinite(sampler.history.ess_target))
    assert all(0 <= value <= 1 for value in sampler.history.mcmc_acceptance)
    assert sampler.n_likelihood_evaluations > 0


@pytest.mark.parametrize("shape", ["flat", "wrong_width"])
def test_smc_rejects_wrong_proposal_event_shape(sampler_kwargs, shape):
    class WrongShapeFlow:
        xp = xp

        def sample_and_log_prob(self, n_samples, context=None, *, rng=None):
            del context, rng
            particles = xp.zeros(
                (n_samples,) if shape == "flat" else (n_samples, 1)
            )
            return particles, xp.zeros(n_samples)

        def log_prob(self, particles, context=None):
            del context
            return xp.zeros(particles.shape[0])

    kwargs = {**sampler_kwargs, "proposal": WrongShapeFlow()}
    sampler = SMCSampler(**kwargs)

    with pytest.raises(ValueError, match=r"must have shape \(n_samples, 2\)"):
        sampler.sample(4, adaptive=False, n_steps=1, rng=4)


def test_aspire_runs_smc_end_to_end(sampler_kwargs):
    aspire = Aspire(
        log_likelihood=sampler_kwargs["log_likelihood"],
        log_prior=sampler_kwargs["log_prior"],
        dims=2,
        parameters=["x", "y"],
        prior_bounds={"x": (-10, 10), "y": (-10, 10)},
        flow=type(sampler_kwargs["proposal"])(),
        xp=xp,
    )

    samples, history = aspire.sample_posterior(
        16,
        sampler="smc",
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"n_steps": 1},
        return_history=True,
    )

    assert isinstance(samples, SMCSamples)
    assert samples.beta == 1.0
    assert isinstance(history, SMCHistory)
    assert history is aspire.sampler.history


def test_smc_can_omit_sample_populations_but_keep_diagnostics(sampler_kwargs):
    sampler = SMCSampler(**sampler_kwargs)
    sampler.sample(
        16,
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"n_steps": 1},
        rng=4,
        store_sample_history=False,
    )

    assert isinstance(sampler.history, SMCHistory)
    assert sampler.history.sample_history == []
    assert len(sampler.history.beta) == 2
    assert len(sampler.history.log_norm_ratio) == 2


def test_smc_mutates_additional_final_samples(sampler_kwargs, monkeypatch):
    final_mutation_steps = []

    def draw_additional_samples(**kwargs):
        final_mutation_steps.append(kwargs["mutation_n_steps"])
        return sequince_draw_additional_samples(**kwargs)

    monkeypatch.setattr(
        "aspire.samplers.smc.draw_additional_samples",
        draw_additional_samples,
    )
    counts = []
    for store_history, n_final_samples in (
        (True, 16),
        (False, 16),
        (True, 40),
        (True, 8),
    ):
        kwargs = dict(sampler_kwargs)
        kwargs["proposal"] = type(sampler_kwargs["proposal"])()
        sampler = SMCSampler(**kwargs)
        samples = sampler.sample(
            16,
            adaptive=False,
            n_steps=2,
            n_final_samples=n_final_samples,
            sampler_kwargs={"n_steps": 1, "n_final_steps": 3},
            rng=4,
            store_sample_history=store_history,
        )
        assert samples.x.shape == (n_final_samples, 2)
        counts.append(sampler.n_likelihood_evaluations)

    assert counts[0] == counts[1]
    assert counts[2] > counts[0]
    assert counts[3] == counts[0]
    assert final_mutation_steps == [3]


def test_minipcn_smc_converts_torch_flow_samples_to_numpy_backend():
    pytest.importorskip("torch")
    from array_api_compat import torch as torch_xp

    class TorchGaussianFlow:
        xp = torch_xp

        def sample_and_log_prob(self, n_samples, context=None, *, rng=None):
            del context
            generator = np.random.default_rng(13) if rng is None else rng
            particles = torch_xp.asarray(generator.normal(size=(n_samples, 2)))
            return particles, self.log_prob(particles)

        def sample(self, n_samples, context=None, *, rng=None):
            return self.sample_and_log_prob(
                n_samples,
                context=context,
                rng=rng,
            )[0]

        def log_prob(self, particles, context=None):
            del context
            return -0.5 * torch_xp.sum(particles**2, axis=-1)

    def numpy_log_prior(samples):
        return -0.5 * xp.sum(samples.x**2, axis=-1)

    def numpy_log_likelihood(samples):
        return -0.5 * xp.sum((samples.x - 1.0) ** 2, axis=-1)

    sampler = MiniPCNSMC(
        log_likelihood=numpy_log_likelihood,
        log_prior=numpy_log_prior,
        dims=2,
        proposal=TorchGaussianFlow(),
        xp=xp,
        parameters=["x", "y"],
    )
    samples = sampler.sample(
        8,
        n_final_samples=12,
        adaptive=False,
        n_steps=1,
        sampler_kwargs={"n_steps": 1},
        rng=7,
    )

    assert isinstance(samples.x, np.ndarray)
    assert samples.x.shape == (12, 2)


def test_smc_supports_native_and_checkpoint_callbacks(sampler_kwargs):
    iterations = []
    checkpoints = []

    def iteration_callback(state, context, result):
        iterations.append((state, context, result))

    sampler = SMCSampler(**sampler_kwargs)
    sampler.sample(
        16,
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"n_steps": 1},
        rng=4,
        callbacks=[iteration_callback],
        checkpoint_callback=checkpoints.append,
        checkpoint_every=1,
    )

    assert len(iterations) == 2
    assert [checkpoint.next_iteration for checkpoint in checkpoints] == [1, 2]
    assert all(
        isinstance(checkpoint, SMCCheckpoint) for checkpoint in checkpoints
    )


def test_aspire_forwards_checkpoint_callback_cadence(sampler_kwargs):
    checkpoints = []
    aspire = Aspire(
        log_likelihood=sampler_kwargs["log_likelihood"],
        log_prior=sampler_kwargs["log_prior"],
        dims=2,
        parameters=["x", "y"],
        prior_bounds={"x": (-10, 10), "y": (-10, 10)},
        flow=type(sampler_kwargs["proposal"])(),
        xp=xp,
    )

    aspire.sample_posterior(
        16,
        sampler="smc",
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"n_steps": 1},
        checkpoint_callback=checkpoints.append,
        checkpoint_every=2,
    )

    assert len(checkpoints) == 1
    assert checkpoints[0].next_iteration == 2


def test_blackjax_options_are_normalized():
    assert SMCSampler._mutation_kwargs(
        "blackjax",
        {
            "algorithm": "nuts",
            "step_size": 0.1,
            "inverse_mass_matrix": [1.0, 1.0],
        },
    ) == {
        "algorithm": "nuts",
        "algorithm_kwargs": {
            "step_size": 0.1,
            "inverse_mass_matrix": [1.0, 1.0],
        },
    }


def test_blackjax_smc_runs():
    pytest.importorskip("blackjax")
    jnp = pytest.importorskip("jax.numpy")

    class JaxGaussianFlow:
        xp = jnp

        def sample_and_log_prob(self, n_samples, context=None, *, rng=None):
            del context, rng
            particles = jnp.reshape(
                jnp.linspace(-1.0, 1.0, n_samples * 2),
                (n_samples, 2),
            )
            return particles, self.log_prob(particles)

        def sample(self, n_samples, context=None, *, rng=None):
            return self.sample_and_log_prob(
                n_samples,
                context=context,
                rng=rng,
            )[0]

        def log_prob(self, particles, context=None):
            del context
            return -0.5 * jnp.sum(particles**2, axis=-1)

    def jax_log_prior(samples):
        return -0.5 * jnp.sum(samples.x**2, axis=-1)

    def jax_log_likelihood(samples):
        return -0.5 * jnp.sum((samples.x - 1.0) ** 2, axis=-1)

    sampler = BlackJAXSMC(
        log_likelihood=jax_log_likelihood,
        log_prior=jax_log_prior,
        dims=2,
        proposal=JaxGaussianFlow(),
        xp=jnp,
        parameters=["x", "y"],
    )
    samples = sampler.sample(
        8,
        n_final_samples=12,
        adaptive=False,
        n_steps=1,
        sampler_kwargs={
            "algorithm": "random-walk",
            "sigma": 0.1,
            "n_steps": 1,
        },
        rng=7,
    )

    assert isinstance(samples.x, pytest.importorskip("jax").Array)
    assert samples.x.shape == (12, 2)
    assert samples.beta == 1.0
    assert len(sampler.history.mcmc_acceptance) == 1


@pytest.mark.parametrize("mutation", ["differential-evolution", "none"])
def test_smc_mutation_override_applies_to_final_samples(
    sampler_kwargs, monkeypatch, mutation: str
) -> None:
    import aspire.samplers.smc as module

    standard_builder = module.make_standard_bayesian_smc
    additional_samples = module.draw_additional_samples
    selected = []

    def build(*args, **kwargs):
        selected.append(kwargs["mutation"])
        return standard_builder(*args, **kwargs)

    def draw(**kwargs):
        selected.append(kwargs["mutation"])
        return additional_samples(**kwargs)

    monkeypatch.setattr(module, "make_standard_bayesian_smc", build)
    monkeypatch.setattr(module, "draw_additional_samples", draw)
    sampler = SMCSampler(**sampler_kwargs)
    samples = sampler.sample(
        16,
        adaptive=False,
        n_steps=2,
        n_final_samples=24,
        mutation=mutation,
        sampler_kwargs={"n_steps": 1},
        rng=13,
    )
    assert selected == [mutation, mutation]
    assert samples.x.shape == (24, 2)
    assert samples.beta == 1.0
    assert sampler.mutation == "differential-evolution"
