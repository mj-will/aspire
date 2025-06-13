import logging
from typing import Callable

import numpy as np

from ...flows.base import Flow
from ...history import SMCHistory
from ...samples import SMCSamples
from ...utils import effective_sample_size, to_numpy, track_calls
from ..base import Sampler

logger = logging.getLogger(__name__)


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

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = 5,
        adaptive: bool = False,
        target_efficiency: float = 0.5,
        n_final_samples: int | None = None,
    ):
        x, log_q = self.flow.sample_and_log_prob(n_samples)
        self.beta = 0.0
        samples = SMCSamples(x, xp=self.xp, log_q=log_q, beta=self.beta)
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )

        if self.xp.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")
        if self.xp.isnan(samples.log_prior).any():
            raise ValueError("Log prior contains NaN values")
        if self.xp.isnan(samples.log_likelihood).any():
            raise ValueError("Log likelihood contains NaN values")

        logger.debug(f"Initial sample summary: {samples}")

        self.history = SMCHistory()

        beta_step = 1 / n_steps
        beta = 0.0
        beta_min = 0.0
        iterations = 0
        while True:
            iterations += 1
            if not adaptive:
                beta += beta_step
                if beta >= 1.0:
                    beta = 1.0
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
                    # Make beta is never larger than 1
                    beta = min(0.5 * (beta_max + beta_min), 1)
            logger.info(f"it {iterations} - beta: {beta}")
            self.history.beta.append(beta)

            ess = effective_sample_size(samples.log_weights(beta))
            self.history.ess.append(ess)
            logger.info(
                f"it {iterations} - ESS: {ess:.1f} ({ess / n_samples:.2f} efficiency)"
            )
            self.history.ess_target.append(
                effective_sample_size(samples.log_weights(1.0))
            )

            log_evidence_ratio = samples.log_evidence_ratio(beta)
            self.history.log_norm_ratio.append(log_evidence_ratio)
            logger.info(
                f"it {iterations} - Log evidence ratio: {log_evidence_ratio}"
            )

            if beta == 1.0:
                if n_final_samples is None:
                    n_final_samples = n_samples
                logger.info(f"Final number of samples: {n_final_samples}")
                samples = samples.resample(beta, n_samples=n_final_samples)
            else:
                samples = samples.resample(beta)

            samples = self.mutate(samples, beta)
            if beta == 1.0:
                break

        samples.log_evidence = samples.xp.sum(self.history.log_norm_ratio)
        samples.log_evidence_error = samples.xp.nan
        final_samples = samples.to_standard_samples()
        logger.info(f"Log evidence: {final_samples.log_evidence:.2f}")
        return final_samples

    def mutate(self, particles):
        raise NotImplementedError

    def log_prob(self, x, beta=None):
        samples = SMCSamples(x, xp=self.xp)
        log_q = self.flow.log_prob(samples.x)
        samples.log_q = samples.array_to_namespace(log_q)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = samples.log_p_t(beta=beta).flatten()
        log_prob[self.xp.isnan(log_prob)] = -self.xp.inf
        return log_prob


class PreconditionedSMC(SMCSampler):
    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        flow: Flow,
        xp: Callable,
        parameters: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(log_likelihood, log_prior, dims, flow, xp, parameters)
        self.pflow = None
        self.pflow_kwargs = kwargs

    def log_prob(self, z, beta=None):
        x, log_j_flow = self.pflow.inverse(z)
        samples = SMCSamples(x, xp=self.xp)
        log_q = self.flow.log_prob(samples.x)
        samples.log_q = samples.array_to_namespace(log_q)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        # Emcee requires numpy arrays
        log_prob = to_numpy(
            samples.log_p_t(beta=beta) + samples.array_to_namespace(log_j_flow)
        ).flatten()
        log_prob[np.isnan(log_prob)] = -np.inf
        return log_prob

    def init_pflow(self):
        FlowClass = self.flow.__class__
        self.pflow = FlowClass(
            dims=self.dims,
            device=self.flow.device,
            data_transform=self.flow.data_transform.new_instance(),
            **self.pflow_kwargs,
        )

    def train_preconditioner(self, samples, **kwargs):
        self.init_pflow()
        self.pflow.fit(samples.x, **kwargs)

    def config_dict(self, include_sample_calls=True):
        config = super().config_dict(include_sample_calls)
        config["preconditioner_kwargs"] = self.pflow_kwargs
        return config
