from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .base import BaseSamples


@dataclass
class NumpySamples(BaseSamples):
    x: np.ndarray
    parameters: list[str] | None = None
    log_likelihood: np.ndarray | None = None
    log_prior: np.ndarray | None = None
    log_q: np.ndarray | None = None
    log_w: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    evidence: float = field(init=False)
    evidence_error: float = field(init=False)
    log_evidence: float = field(init=False)
    log_evidence_error: float = field(init=False)
    effective_sample_size: float = field(init=False)

    def compute_weights(self):
        self.log_w = self.log_likelihood + self.log_prior - self.log_q
        self.log_evidence = logsumexp(self.log_w) - np.log(len(self.x))
        self.weights = np.exp(self.log_w)
        self.evidence = np.exp(self.log_evidence)
        n = len(self.x)
        self.evidence_error = np.sqrt(
            np.sum((self.weights - self.evidence) ** 2) / (n * (n - 1))
        )
        self.log_evidence_error = abs(self.evidence_error / self.evidence)
        self.effective_sample_size = np.exp(
            logsumexp(self.log_w) * 2 - logsumexp(self.log_w * 2)
        )

    @property
    def scaled_weights(self):
        return np.exp(self.log_w - self.log_w.max())

    def rejection_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        log_u = np.log(rng.uniform(size=len(self.x)))
        log_w = self.log_w - self.log_w.max()
        accept = log_w > log_u
        return self.__class__(
            x=self.x[accept],
            log_likelihood=self.log_likelihood[accept],
            log_prior=self.log_prior[accept],
        )

    def to_numpy(self):
        return self

    def to_jax(self):
        from .jax import JaxSamples, numpy_to_jax

        return JaxSamples(
            x=numpy_to_jax(self.x),
            log_likelihood=numpy_to_jax(self.log_likelihood),
            log_prior=numpy_to_jax(self.log_prior),
            log_q=numpy_to_jax(self.log_q),
        )

    def to_torch(self):
        from .torch import TorchSamples, numpy_to_torch

        return TorchSamples(
            x=numpy_to_torch(self.x),
            log_likelihood=numpy_to_torch(self.log_likelihood),
            log_prior=numpy_to_torch(self.log_prior),
            log_q=numpy_to_torch(self.log_q),
        )


def torch_to_numpy(value, /):
    return value.detach().numpy() if value is not None else None


def jax_to_numpy(value, /):
    return np.array(value) if value is not None else None
