import logging
from typing import Callable

import array_api_compat.numpy as np

from ...flows.base import Flow
from ...history import SMCHistory
from ...samples import SMCSamples
from ...utils import (
    asarray,
    effective_sample_size,
    track_calls,
    update_at_indices,
)
from ..mcmc import MCMCSampler

logger = logging.getLogger(__name__)


class SMCSampler(MCMCSampler):
    """Base class for Sequential Monte Carlo samplers."""

    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        prior_flow: Flow,
        xp: Callable,
        parameters: list[str] | None = None,
        preconditioning_transform: Callable | None = None,
    ):
        super().__init__(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            dims=dims,
            prior_flow=prior_flow,
            xp=xp,
            parameters=parameters,
            preconditioning_transform=preconditioning_transform,
        )

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = None,
        adaptive: bool = True,
        target_efficiency: float = 0.5,
        n_final_samples: int | None = None,
    ):
        samples = self.draw_initial_samples(n_samples)
        samples = SMCSamples.from_samples(samples, xp=self.xp, beta=0.0)
        self.fit_preconditioning_transform(samples.x)

        if self.xp.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")
        if self.xp.isnan(samples.log_prior).any():
            raise ValueError("Log prior contains NaN values")
        if self.xp.isnan(samples.log_likelihood).any():
            raise ValueError("Log likelihood contains NaN values")

        logger.debug(f"Initial sample summary: {samples}")

        self.history = SMCHistory()

        if n_steps is not None:
            beta_step = 1 / n_steps
        elif not adaptive:
            raise ValueError("Either n_steps or adaptive=True must be set")
        else:
            beta_step = np.nan
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
                if self.xp.isnan(eff):
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

        samples.log_evidence = samples.xp.sum(
            asarray(self.history.log_norm_ratio, self.xp)
        )
        samples.log_evidence_error = samples.xp.nan
        final_samples = samples.to_standard_samples()
        logger.info(f"Log evidence: {final_samples.log_evidence:.2f}")
        return final_samples

    def mutate(self, particles):
        raise NotImplementedError

    def log_prob(self, z, beta=None):
        x, log_abs_det_jacobian = self.preconditioning_transform.inverse(z)
        samples = SMCSamples(x, xp=self.xp)
        log_q = self.prior_flow.log_prob(samples.x)
        samples.log_q = samples.array_to_namespace(log_q)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = samples.log_p_t(
            beta=beta
        ).flatten() + samples.array_to_namespace(log_abs_det_jacobian)

        log_prob = update_at_indices(
            log_prob, self.xp.isnan(log_prob), -self.xp.inf
        )
        return log_prob


class NumpySMCSampler(SMCSampler):
    def __init__(
        self,
        log_likelihood,
        log_prior,
        dims,
        prior_flow,
        xp,
        parameters=None,
        preconditioning_transform=None,
    ):
        if preconditioning_transform is not None:
            preconditioning_transform = preconditioning_transform.new_instance(
                xp=np
            )
        super().__init__(
            log_likelihood,
            log_prior,
            dims,
            prior_flow,
            xp,
            parameters=parameters,
            preconditioning_transform=preconditioning_transform,
        )
