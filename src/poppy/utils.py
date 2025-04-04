from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any

import array_api_compat.numpy as np
from array_api_compat import array_namespace, is_torch_namespace

if TYPE_CHECKING:
    from multiprocessing import Pool

    from array_api_compat.common._typing import Array

    from .poppy import Poppy

logger = logging.getLogger(__name__)


def configure_logger(log_level: str | int = "INFO") -> logging.Logger:
    """Configure the logger.

    Adds a stream handler to the logger.
    """
    logger = logging.getLogger("poppy")
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class PoolHandler:
    """Context manager to temporarily replace the log_likelihood method of a
    Poppy instance with a version that uses a multiprocessing pool to
    parallelize computation.

    Parameters
    ----------
    poppy_instance : Poppy
        The Poppy instance to modify. The log_likelihood method of this
        instance must accept a 'map_fn' keyword argument.
    pool : multiprocessing.Pool
        The pool to use for parallel computation.
    """

    def __init__(self, poppy_instance: Poppy, pool: Pool):
        self.poppy_instance = poppy_instance
        self.pool = pool

    @property
    def poppy_instance(self):
        return self._poppy_instance

    @poppy_instance.setter
    def poppy_instance(self, value: Poppy):
        signature = inspect.signature(value.log_likelihood)
        if "map_fn" not in signature.parameters:
            raise ValueError(
                "The log_likelihood method of the Poppy instance must accept a"
                " 'map_fn' keyword argument."
            )
        self._poppy_instance = value

    def __enter__(self):
        logger.debug("Updating map function in log-likelihood method")
        self.original_log_likelihood = self.poppy_instance.log_likelihood
        self.poppy_instance.log_likelihood = partial(
            self.original_log_likelihood, map_fn=self.pool.map
        )
        return self.pool

    def __exit__(self, exc_type, exc_value, traceback):
        self.poppy_instance.log_likelihood = self.original_log_likelihood
        self.pool.close()
        self.pool.join()


def logit(x: Array, eps: float | None = None) -> tuple[Array, Array]:
    """Logit function that also returns log Jacobian determinant.

    Parameters
    ----------
    x : float or ndarray
        Array of values
    eps : float, optional
        Epsilon value used to clamp inputs to [eps, 1 - eps]. If None, then
        inputs are not clamped.

    Returns
    -------
    float or ndarray
        Rescaled values.
    float or ndarray
        Log Jacobian determinant.
    """
    xp = array_namespace(x)
    if eps:
        x = xp.clip(x, eps, 1 - eps)
    y = xp.log(x) - xp.log1p(-x)
    log_j = (-xp.log(x) - xp.log1p(-x)).sum(-1)
    return y, log_j


def sigmoid(x: Array) -> tuple[Array, Array]:
    """Sigmoid function that also returns log Jacobian determinant.

    Parameters
    ----------
    x : float or ndarray
        Array of values

    Returns
    -------
    float or ndarray
        Rescaled values.
    float or ndarray
        Log Jacobian determinant.
    """
    xp = array_namespace(x)
    x = xp.divide(1, 1 + xp.exp(-x))
    log_j = (xp.log(x) + xp.log1p(-x)).sum(-1)
    return x, log_j


def logsumexp(x: Array, axis: int | None = None) -> Array:
    """Implementation of logsumexp that works with array api.

    This will be removed once the implementation in scipy is compatible.
    """
    xp = array_namespace(x)
    c = x.max()
    return c + xp.log(xp.sum(xp.exp(x - c), axis=axis))


def to_numpy(x: Array) -> np.ndarray:
    """Convert an array to a numpy array.

    This automatically moves the device to the CPU.
    """
    return np.asarray(np.to_device(x, "cpu"))


def effective_sample_size(log_w: Array) -> float:
    xp = array_namespace(log_w)
    return xp.exp(xp.asarray(logsumexp(log_w) * 2 - logsumexp(log_w * 2)))


@contextmanager
def disable_gradients(xp, inference: bool = True):
    """Disable gradients for a specific array API.

    Usage:

    ```python
    with disable_gradients(xp):
        # Do something
    ```

    Parameters
    ----------
    xp : module
        The array API module to use.
    inference : bool, optional
        When using PyTorch, set to True to enable inference mode.
    """
    if is_torch_namespace(xp):
        if inference:
            with xp.inference_mode():
                yield
        else:
            with xp.no_grad():
                yield
    else:
        yield


def encode_for_hdf5(value: Any) -> Any:
    """Encode a value for storage in an HDF5 file."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, dict):
        return {k: encode_for_hdf5(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_for_hdf5(v) for v in value]
    if isinstance(value, set):
        return {encode_for_hdf5(v) for v in value}
    return value


def update_at_indices(x: Array, slc: Array, y: Array) -> Array:
    """Update an array at specific indices."

    This is a workaround for the fact that array API does not support
    advanced indexing with all backends.

    Parameters
    ----------
    x : Array
        The array to update.
    slc : Array
        The indices to update.
    y : Array
        The values to set at the indices.
    
    Returns
    -------
    Array
        The updated array.
    """
    try:
        x[slc] = y
        return x
    except TypeError:
        return x.at[slc].set(y)
