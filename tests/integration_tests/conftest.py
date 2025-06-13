import  numpy as np
import pytest
from poppy.samples import Samples


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
    return Samples(initial_samples, xp=xp)


@pytest.fixture(params=[True, False])
def bounded_to_unbounded(request):
    return request.param


@pytest.fixture(params=["zuko", "flowjax"])
def flow_backend(request):
    return request.param


@pytest.fixture(params=["jax", "numpy", "torch"])
def samples_backend(request):
    return request.param


@pytest.fixture
def xp(samples_backend):
    if samples_backend == "jax":
        import jax.numpy as xp
    elif samples_backend == "torch":
        import array_api_compat.torch as xp
    elif samples_backend == "numpy":
        import array_api_compat.numpy as xp
    else:
        raise ValueError(f"Unsupported backend: {samples_backend}")
    return xp


@pytest.fixture
def log_likelihood(likelihood_mean, likelihood_std):
    xp = np
    def _log_likelihood(samples):
        x = xp.asarray(samples.x)
        constant = xp.log(1 / (likelihood_std * xp.sqrt(2 * xp.pi)))
        return xp.sum(constant - (0.5 * ((x - likelihood_mean) / likelihood_std) ** 2), axis=-1)
    return _log_likelihood



@pytest.fixture
def log_prior(dims):
    xp = np
    def _log_prior(samples):
        x = xp.asarray(samples.x)
        constant = dims * xp.log(1 / 10)
        val = xp.where((x >= -10) & (x <= 10), constant, -xp.inf)
        return xp.sum(val, axis=-1)
    return _log_prior


