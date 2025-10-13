import numpy as np
import pytest
from array_api_compat import is_torch_namespace


@pytest.fixture(autouse=True, scope="session")
def enable_jax_float64():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass


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
    return xp.dtype(request.param)
