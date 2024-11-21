from functools import partial
import inspect
import logging
import multiprocessing as mp

from .poppy import Poppy

logger = logging.getLogger(__name__)


def configure_logger(log_level: str | int = "INFO") -> logging.Logger:
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
    def __init__(self, poppy_instance: Poppy, pool: mp.Pool):
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
        logger.info("Updating map function in log-likelihood method")
        self.original_log_likelihood = self.poppy_instance.log_likelihood
        self.poppy_instance.log_likelihood = partial(
            self.original_log_likelihood, map_fn=self.pool.map
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.poppy_instance.log_likelihood = self.original_log_likelihood
