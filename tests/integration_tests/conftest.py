import math
from collections import namedtuple

import pytest

from aspire.samples import Samples

SamplerConfig = namedtuple(
    "SamplerConfig",
    ["sampler", "sampler_kwargs"],
)


@pytest.fixture
def dims():
    return 2


@pytest.fixture
def parameters(dims):
    return [f"x_{i}" for i in range(dims)]


@pytest.fixture
def prior_bounds(parameters):
    return {p: [-10, 10] for p in parameters}


@pytest.fixture
def n_samples():
    return 500


@pytest.fixture
def likelihood_mean():
    return 2.0


@pytest.fixture
def likelihood_std():
    return 1.0


@pytest.fixture
def initial_samples(dims, rng, likelihood_mean, likelihood_std, n_samples):
    return rng.normal(likelihood_mean, likelihood_std, size=(n_samples, dims))


@pytest.fixture
def samples(initial_samples, xp):
    assert initial_samples.ndim == 2
    return Samples(initial_samples, xp=xp)


@pytest.fixture(params=[True, False])
def bounded_to_unbounded(request):
    return request.param


@pytest.fixture(params=["zuko", "flowjax"])
def flow_backend(request):
    if request.param == "zuko":
        pytest.importorskip("torch")
        pytest.importorskip("zuko")
    else:
        pytest.importorskip("jax")
        pytest.importorskip("flowjax")
    return request.param


@pytest.fixture(params=["jax", "numpy", "torch"])
def samples_backend(request):
    return request.param


@pytest.fixture
def xp(samples_backend):
    if samples_backend == "jax":
        xp = pytest.importorskip("jax.numpy")
    elif samples_backend == "torch":
        pytest.importorskip("torch")
        xp = pytest.importorskip("array_api_compat.torch")
    elif samples_backend == "numpy":
        import array_api_compat.numpy as xp
    else:
        raise ValueError(f"Unsupported backend: {samples_backend}")
    return xp


@pytest.fixture
def log_likelihood(likelihood_mean, likelihood_std, xp, dims):
    def _log_likelihood(samples):
        assert samples.x.shape[-1] == dims
        x = xp.asarray(samples.x)
        constant = xp.log(
            xp.asarray(1 / (likelihood_std * math.sqrt(2 * math.pi)))
        )
        return xp.sum(
            constant - (0.5 * ((x - likelihood_mean) / likelihood_std) ** 2),
            axis=-1,
        )

    return _log_likelihood


@pytest.fixture
def log_prior(dims, xp):
    def _log_prior(samples):
        assert samples.x.shape[-1] == dims
        x = xp.asarray(samples.x)
        constant = dims * xp.log(xp.asarray(1 / 10))
        val = xp.where((x >= -10) & (x <= 10), constant, -xp.inf)
        return xp.sum(val, axis=-1)

    return _log_prior


@pytest.fixture(
    params=[
        "importance",
        "minipcn_smc",
        "emcee_smc",
        "smc",
        "emcee",
        "minipcn",
        "blackjax_smc_rwmh",
        "blackjax_smc_nuts",
    ]
)
def sampler_config(request):
    if request.param == "importance":
        return SamplerConfig(sampler="importance", sampler_kwargs={})
    elif request.param == "emcee":
        return SamplerConfig(
            sampler="emcee",
            sampler_kwargs={
                # This matrix exercises adapters and persistence. Dedicated
                # Gaussian tests use longer chains to check sampling accuracy.
                "nsteps": 5,
                "vectorize": True,
            },
        )
    elif request.param == "minipcn":
        return SamplerConfig(
            sampler="minipcn",
            sampler_kwargs={
                "n_steps": 5,
            },
        )
    elif request.param in ["smc", "minipcn_smc"]:
        return SamplerConfig(
            sampler="minipcn_smc",
            sampler_kwargs={
                "adaptive": True,
                "sampler_kwargs": {
                    "n_steps": 10,
                },
            },
        )
    elif request.param == "emcee_smc":
        return SamplerConfig(
            sampler="emcee_smc",
            sampler_kwargs={
                "adaptive": True,
                "sampler_kwargs": {
                    "nsteps": 10,
                    "progress": False,
                    "vectorize": True,
                },
            },
        )
    elif request.param == "blackjax_smc_rwmh":
        return SamplerConfig(
            sampler="blackjax_smc",
            sampler_kwargs={
                "adaptive": True,
                "sampler_kwargs": {
                    "algorithm": "random-walk",
                    "algorithm_kwargs": {"sigma": 0.1},
                    "n_steps": 10,
                },
            },
        )
    elif request.param == "blackjax_smc_nuts":
        return SamplerConfig(
            sampler="blackjax_smc",
            sampler_kwargs={
                "adaptive": True,
                "sampler_kwargs": {
                    "algorithm": "nuts",
                    "algorithm_kwargs": {"step_size": 0.1},
                    "n_steps": 10,
                },
            },
        )
    else:
        raise ValueError(f"Unsupported sampler: {request.param}")
