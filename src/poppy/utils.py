from __future__ import annotations

from functools import partial
import inspect
import logging
from array_api_compat import array_namespace
from typing import TYPE_CHECKING
import array_api_compat.numpy as np

if TYPE_CHECKING:
    from array_api_compat.common._typing import Array
    from multiprocessing import Pool
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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

    def __exit__(self, exc_type, exc_value, traceback):
        self.poppy_instance.log_likelihood = self.original_log_likelihood



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
    log_j = -xp.log(y * (1 - y)).sum(-1)
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
    log_j = xp.log(x * (1 - x)).sum(-1)
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
    return xp.exp(
        xp.asarray(logsumexp(log_w) * 2 - logsumexp(log_w * 2))
    )