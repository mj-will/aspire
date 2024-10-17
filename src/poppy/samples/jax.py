from dataclasses import dataclass, field

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .base import BaseSamples


@dataclass
class JaxSamples(BaseSamples):
    x: jnp.ndarray
    log_likelihood: jnp.ndarray | None = None
    log_prior: jnp.ndarray | None = None
    log_q: jnp.ndarray | None = None
    log_w: jnp.ndarray = field(init=False)
    weights: jnp.ndarray = field(init=False)
    evidence: float = field(init=False)
    evidence_error: float = field(init=False)
    log_evidence: float = field(init=False)
    log_evidence_error: float = field(init=False)
    effective_sample_size: float = field(init=False)

    def compute_weights(self):
        self.log_w = self.log_likelihood + self.log_prior - self.log_q
        self.log_evidence = logsumexp(self.log_w) - jnp.log(len(self.x))
        self.weights = jnp.exp(self.log_w - jnp.max(self.log_w))
        self.evidence = jnp.exp(self.log_evidence)
        n = len(self.x)
        self.evidence_error = jnp.sqrt(
            jnp.sum((self.weights - self.evidence) ** 2) / (n * (n - 1))
        )
        self.log_evidence_error = jnp.abs(self.evidence_error / self.evidence)
        self.effective_sample_size = jnp.exp(
            logsumexp(self.log_w) * 2 - logsumexp(self.log_w * 2)
        )

    @staticmethod
    def combine_samples(*samples):
        x = jnp.concatenate([sample.x for sample in samples], axis=0)
        if any(sample.log_likelihood is None for sample in samples):
            log_likelihood = None
        else:
            log_likelihood = jnp.concatenate(
                [sample.log_likelihood for sample in samples], axis=0
            )
        if any(sample.log_prior is None for sample in samples):
            log_prior = None
        else:
            log_prior = jnp.concatenate(
                [sample.log_prior for sample in samples], axis=0
            )
        if any(sample.log_q is None for sample in samples):
            log_q = None
        else:
            log_q = jnp.concatenate(
                [sample.log_q for sample in samples], axis=0
            )
        return JaxSamples(x, log_likelihood, log_prior, log_q)

    def to_jax(self):
        return self

    def to_numpy(self):
        from .numpy import NumpySamples, jax_to_numpy

        return NumpySamples(
            jax_to_numpy(self.x),
            log_likelihood=jax_to_numpy(self.log_likelihood),
            log_prior=jax_to_numpy(self.log_prior),
            log_q=jax_to_numpy(self.log_q),
        )


def numpy_to_jax(value, /):
    return jnp.array(value) if value is not None else None


def torch_to_jax(value, /):
    return jnp.array(value.detach().numpy()) if value is not None else None
