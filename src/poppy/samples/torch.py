import math
from dataclasses import dataclass, field

import torch

from .base import BaseSamples


@dataclass
class TorchSamples(BaseSamples):
    x: torch.Tensor
    log_likelihood: torch.Tensor | None = None
    log_prior: torch.Tensor | None = None
    log_q: torch.Tensor | None = None
    log_w: torch.Tensor = field(init=False)
    weights: torch.Tensor = field(init=False)
    evidence: float = field(init=False)
    evidence_error: float = field(init=False)
    log_evidence: float = field(init=False)
    log_evidence_error: float = field(init=False)
    effective_sample_size: float = field(init=False)

    def compute_weights(self):
        self.log_w = self.log_likelihood + self.log_prior - self.log_q
        self.log_evidence = torch.logsumexp(self.log_w, 0) - math.log(
            len(self.x)
        )
        self.weights = torch.exp(self.log_w - torch.max(self.log_w))
        self.evidence = torch.exp(self.log_evidence)
        n = len(self.x)
        self.evidence_error = torch.sqrt(
            torch.sum((self.weights - self.evidence) ** 2) / (n * (n - 1))
        )
        self.log_evidence_error = torch.abs(
            self.evidence_error / self.evidence
        )
        self.effective_sample_size = torch.exp(
            torch.logsumexp(self.log_w, 0) * 2
            - torch.logsumexp(self.log_w * 2, 0)
        )

    def to_numpy(self):
        from .numpy import NumpySamples, torch_to_numpy

        return NumpySamples(
            torch_to_numpy(self.x),
            parameters=self.parameters,
            log_likelihood=torch_to_numpy(self.log_likelihood),
            log_prior=torch_to_numpy(self.log_prior),
            log_q=torch_to_numpy(self.log_q),
        )

    def to_torch(self):
        return self

    def to_jax(self):
        from .jax import JaxSamples, torch_to_jax

        return JaxSamples(
            x=torch_to_jax(self.x),
            parameters=self.parameters,
            log_likelihood=torch_to_jax(self.log_likelihood),
            log_prior=torch_to_jax(self.log_prior),
            log_q=torch_to_jax(self.log_q),
        )


def numpy_to_torch(value, /, device=None, dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    return (
        torch.tensor(value, dtype=dtype, device=device)
        if value is not None
        else None
    )


def jax_to_torch(value, /, device=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    return (
        torch.tensor(value, dtype=dtype, device=device)
        if value is not None
        else None
    )
