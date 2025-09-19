import logging
from typing import Callable

import jax.numpy as jnp
import jax.random as jrandom
from flowjax.train import fit_to_data

from ..base import Flow
from .utils import get_flow

logger = logging.getLogger(__name__)


class FlowJax(Flow):
    xp = jnp

    def __init__(self, dims: int, key=None, data_transform=None, **kwargs):
        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("The device argument is not used in FlowJax. ")
        super().__init__(dims, device=device, data_transform=data_transform)
        if key is None:
            key = jrandom.key(0)
            logger.warning(
                "The key argument is None. "
                "A random key will be used for the flow. "
                "Results may not be reproducible."
            )
        self.key = key
        self.loc = None
        self.scale = None
        self.key, subkey = jrandom.split(self.key)
        self._flow = get_flow(
            key=subkey,
            dims=self.dims,
            **kwargs,
        )

    def fit(self, x, **kwargs):
        from ...history import FlowHistory

        x = jnp.asarray(x)
        x_prime = self.fit_data_transform(x)
        self.key, subkey = jrandom.split(self.key)
        self._flow, losses = fit_to_data(subkey, self._flow, x_prime, **kwargs)
        return FlowHistory(
            training_loss=list(losses["train"]),
            validation_loss=list(losses["val"]),
        )

    def forward(self, x, xp: Callable = jnp):
        x_prime, log_abs_det_jacobian = self.rescale(x)
        z, log_abs_det_jacobian_flow = self._flow.forward(x_prime)
        return xp.asarray(z), xp.asarray(
            log_abs_det_jacobian + log_abs_det_jacobian_flow
        )

    def inverse(self, z, xp: Callable = jnp):
        z = jnp.asarray(z)
        x_prime, log_abs_det_jacobian_flow = self._flow.inverse(z)
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(
            log_abs_det_jacobian + log_abs_det_jacobian_flow
        )

    def log_prob(self, x, xp: Callable = jnp):
        x_prime, log_abs_det_jacobian = self.rescale(x)
        log_prob = self._flow.log_prob(x_prime)
        return xp.asarray(log_prob + log_abs_det_jacobian)

    def sample(self, n_samples: int, xp: Callable = jnp):
        self.key, subkey = jrandom.split(self.key)
        x_prime = self._flow.sample(subkey, (n_samples,))
        x = self.inverse_rescale(x_prime)[0]
        return xp.asarray(x)

    def sample_and_log_prob(self, n_samples: int, xp: Callable = jnp):
        self.key, subkey = jrandom.split(self.key)
        x_prime = self._flow.sample(subkey, (n_samples,))
        log_prob = self._flow.log_prob(x_prime)
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(log_prob - log_abs_det_jacobian)
