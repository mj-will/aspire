import numpy as np
import pytest
from array_api_compat import is_jax_namespace, is_torch_namespace


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture(params=["jax", "torch", "numpy"])
def xp(request):
    if request.param == "jax":
        import jax.numpy as xp
    elif request.param == "torch":
        import array_api_compat.torch as xp
    elif request.param == "numpy":
        import array_api_compat.numpy as xp
    else:
        raise ValueError(f"Unsupported backend: {request.param}")
    return xp


@pytest.fixture(params=["float32", "float64", None])
def dtype(request, xp):
    if request.param is None:
        return None
    if is_torch_namespace(xp):
        import torch

        if request.param == "float32":
            return torch.float32
        elif request.param == "float64":
            return torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {request.param}")
    elif is_jax_namespace(xp) and request.param == "float64":
        # Skip float64 tests for JAX since you can't change dtype like this
        pytest.skip("JAX does not support float64 by default.")
        return

    return xp.dtype(request.param)
