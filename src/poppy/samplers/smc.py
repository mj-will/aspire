import logging
from dataclasses import dataclass
from typing import Callable

import emcee
import numpy as np
from array_api_compat.common._typing import Array

from ..flows.base import Flow
from ..history import SMCHistory
from ..samples import BaseSamples
from ..utils import effective_sample_size, logsumexp, to_numpy
from .base import Sampler

logger = logging.getLogger(__name__)


@dataclass
class SMCSamples(BaseSamples):
    beta: float | None = None
    """Temperature parameter for the current samples."""

    def log_p_t(self, beta):
        log_p_T = self.log_likelihood + self.log_prior
        return (1 - beta) * self.log_q + beta * log_p_T

    def unnormalized_log_weights(self, beta):
        return (self.beta - beta) * self.log_q + (beta - self.beta) * (
            self.log_likelihood + self.log_prior
        )

    def log_evidence_ratio(self, beta):
        log_w = self.unnormalized_log_weights(beta)
        return logsumexp(log_w) - self.xp.log(len(self.x))

    def log_weights(self, beta) -> Array:
        log_w = self.unnormalized_log_weights(beta)
        if self.xp.isnan(log_w).any():
            raise ValueError(f"Log weights contain NaN values for beta={beta}")
        log_evidence_ratio = logsumexp(log_w) - self.xp.log(len(self.x))
        return log_w + log_evidence_ratio

    def resample(self, beta) -> "SMCSamples":
        if beta == self.beta:
            logger.warning("Resampling with the same beta value")
            return self
        log_w = self.log_weights(beta)
        w = self.xp.exp(log_w - logsumexp(log_w))
        idx = np.random.choice(
            len(self.x), size=len(self.x), replace=True, p=w
        )
        return self.__class__(
            x=self.x[idx],
            log_likelihood=self.log_likelihood[idx],
            log_prior=self.log_prior[idx],
            log_q=self.log_q[idx],
            beta=beta,
        )


class SMCSampler(Sampler):
    """Base class for Sequential Monte Carlo samplers."""

    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        flow: Flow,
        xp: Callable,
        parameters: list[str] | None = None,
    ):
        super().__init__(log_likelihood, log_prior, dims, flow, xp, parameters)

    def sample(self, n_samples: int):
        raise NotImplementedError

    def mutate(self, particles):
        raise NotImplementedError

    def log_prob(self, x, beta=None):
        samples = SMCSamples(x, xp=self.xp)
        log_q = self.flow.log_prob(samples.x)
        samples.log_q = samples.array_to_namespace(log_q)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = to_numpy(samples.log_p_t(beta=beta)).flatten()
        log_prob[self.xp.isnan(log_prob)] = -self.xp.inf
        return log_prob


class EmceeSMC(SMCSampler):
    def sample(
        self,
        n_samples: int,
        n_steps: int = 5,
        adaptive: bool = False,
        target_efficiency: float = 0.5,
        emcee_kwargs: dict | None = None,
    ):
        self.emcee_kwargs = emcee_kwargs or {}
        self.emcee_kwargs.setdefault("nsteps", 50)
        self.emcee_kwargs.setdefault("progress", True)

        x, log_q = self.flow.sample_and_log_prob(n_samples)
        self.beta = 0.0
        samples = SMCSamples(x, xp=self.xp, log_q=log_q, beta=self.beta)
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )

        logger.debug(f"Initial sample summary: {samples}")

        history = SMCHistory()

        beta_step = 1 / n_steps
        beta = 0.0
        beta_min = 0.0
        iterations = 0
        while True:
            iterations += 1
            if not adaptive:
                beta += beta_step
            else:
                beta_max = 1.0
                ess = effective_sample_size(samples.log_weights(beta_max))
                eff = ess / len(samples.x)
                if np.isnan(eff):
                    raise ValueError("Effective sample size is NaN")
                beta = beta_max
                while True:
                    ess = effective_sample_size(samples.log_weights(beta))
                    eff = ess / n_samples
                    if eff >= target_efficiency:
                        beta_min = beta
                        break
                    else:
                        beta_max = beta
                    beta = 0.5 * (beta_max + beta_min)
            logger.info(f"it {iterations} - beta: {beta}")
            history.beta.append(beta)

            ess = effective_sample_size(samples.log_weights(beta))
            history.ess.append(ess)
            logger.info(
                f"it {iterations} - ESS: {ess:.1f} ({ess / n_samples:.2f} efficiency)"
            )
            history.ess_target.append(
                effective_sample_size(samples.log_weights(1.0))
            )

            log_evidence_ratio = samples.log_evidence_ratio(beta)
            history.log_norm_ratio.append(log_evidence_ratio)
            logger.info(
                f"it {iterations} - Log evidence ratio: {log_evidence_ratio}"
            )

            samples = samples.resample(beta)
            samples = self.mutate(samples, beta)
            if beta == 1.0:
                break

        self.history = history
        samples.log_q = None
        logger.info(
            f"Likelihood evaluations: {iterations * n_samples * self.emcee_kwargs['nsteps']}"
        )
        return samples

    def mutate(self, particles, beta):
        logger.info("Mutating particles")
        sampler = emcee.EnsembleSampler(
            len(particles.x),
            self.dims,
            self.log_prob,
            args=(beta,),
            vectorize=True,
        )
        sampler.run_mcmc(to_numpy(particles.x), **self.emcee_kwargs)
        x = sampler.get_chain(flat=False)[-1, ...]
        samples = SMCSamples(x, xp=self.xp, beta=beta)
        samples.log_q = samples.array_to_namespace(
            self.flow.log_prob(samples.x)
        )
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        if np.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")
        return samples
