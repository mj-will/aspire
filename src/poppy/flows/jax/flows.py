import jax.numpy as jnp
import jax.random as jrandom
from flowjax.train import fit_to_data

from ..base import Flow
from .utils import get_flow


class FlowJax(Flow):
    def __init__(self, dims: int, key, **kwargs):
        super().__init__(dims)
        self.key = key
        self.loc = None
        self.scale = None
        self.key, subkey = jrandom.split(self.key)
        self._flow = get_flow(
            key=subkey,
            dims=self.dims,
            **kwargs,
        )

    def fit(self, x):
        from ...history import History

        self.fit_initial_transforms(x)
        x_prime, _ = self.rescale(x)
        self.key, subkey = jrandom.split(self.key)
        self._flow, losses = fit_to_data(subkey, self._flow, x_prime)
        return History(
            training_loss=list(map(lambda x: x.item(), losses["train"])),
            validation_loss=list(map(lambda x: x.item(), losses["val"])),
        )

    def sample(self, n_samples: int):
        self.key, subkey = jrandom.split(self.key)
        x_prime = self._flow.sample(subkey, (n_samples,))
        log_prob = self._flow.log_prob(x_prime)
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return x, log_prob + log_abs_det_jacobian

    def fit_initial_transforms(self, x: jnp.ndarray) -> None:
        self.loc = jnp.mean(x, axis=0)
        self.scale = jnp.std(x, axis=0)
        self.log_abs_det_jacobian = jnp.sum(jnp.log(self.scale))

    def rescale(self, x):
        return (
            x - self.loc
        ) / self.scale, self.log_abs_det_jacobian * jnp.ones(x.shape[0])

    def inverse_rescale(self, x_prime):
        return (
            x_prime * self.scale + self.loc,
            -self.log_abs_det_jacobian * jnp.ones(x_prime.shape[0]),
        )
