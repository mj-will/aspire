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
        rng: np.random.Generator | None = None,
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
        self.rng = rng or np.random.default_rng()
        self._adapative_target_efficiency = False

    @property
    def target_efficiency(self):
        return self._target_efficiency

    @target_efficiency.setter
    def target_efficiency(self, value: float | tuple):
        """Set the target efficiency.

        Parameters
        ----------
        value : float or tuple
            If a float, the target efficiency to use for all iterations.
            If a tuple of two floats, the target efficiency will adapt from
            the first value to the second value over the course of the SMC
            iterations. See `target_efficiency_rate` for details.
        """
        if isinstance(value, float):
            if not (0 < value < 1):
                raise ValueError("target_efficiency must be in (0, 1)")
            self._target_efficiency = value
            self._adapative_target_efficiency = False
        elif len(value) != 2:
            raise ValueError(
                "target_efficiency must be a float or tuple of two floats"
            )
        else:
            value = tuple(map(float, value))
            if not (0 < value[0] < value[1] < 1):
                raise ValueError(
                    "target_efficiency tuple must be in (0, 1) and increasing"
                )
            self._target_efficiency = value
            self._adapative_target_efficiency = True

    def current_target_efficiency(self, beta: float) -> float:
        """Get the current target efficiency based on beta."""
        if self._adapative_target_efficiency:
            return self._target_efficiency[0] + (
                self._target_efficiency[1] - self._target_efficiency[0]
            ) * (beta**self.target_efficiency_rate)
        else:
            return self._target_efficiency

    def determine_beta(
        self,
        samples: SMCSamples,
        beta: float,
        beta_step: float,
        min_step: float,
    ) -> tuple[float, float]:
        """Determine the next beta value.

        Parameters
        ----------
        samples : SMCSamples
            The current samples.
        beta : float
            The current beta value.
        beta_step : float
            The fixed beta step size if not adaptive.
        min_step : float
            The minimum beta step size.

        Returns
        -------
        beta : float
            The new beta value.
        min_step : float
            The new minimum step size if adaptive_min_step is True.
        """
        if not self.adaptive:
            beta += beta_step
            if beta >= 1.0:
                beta = 1.0
        else:
            beta_prev = beta
            beta_min = beta_prev
            beta_max = 1.0
            tol = 1e-5
            eff_beta_max = effective_sample_size(
                samples.log_weights(beta_max)
            ) / len(samples)
            if eff_beta_max >= self.current_target_efficiency(beta_prev):
                beta_min = 1.0
            target_eff = self.current_target_efficiency(beta_prev)
            while beta_max - beta_min > tol:
                beta_try = 0.5 * (beta_max + beta_min)
                eff = effective_sample_size(
                    samples.log_weights(beta_try)
                ) / len(samples)
                if eff >= target_eff:
                    beta_min = beta_try
                else:
                    beta_max = beta_try
            beta_star = beta_min

            if self.adaptive_min_step:
                min_step = min_step * (1 - beta_prev) / (1 - beta_star)
            beta = max(beta_star, beta_prev + min_step)
            beta = min(beta, 1.0)
        return beta, min_step

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int | None = None,
        adaptive: bool = True,
        min_step: float | None = None,
        max_n_steps: int | None = None,
        target_efficiency: float = 0.5,
        target_efficiency_rate: float = 1.0,
        n_final_samples: int | None = None,
    ) -> SMCSamples:
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

        # Remove the n_final_steps from sampler_kwargs if present
        self.sampler_kwargs = self.sampler_kwargs or {}
        n_final_steps = self.sampler_kwargs.pop("n_final_steps", None)

        self.history = SMCHistory()

        self.target_efficiency = target_efficiency
        self.target_efficiency_rate = target_efficiency_rate

        if n_steps is not None:
            beta_step = 1 / n_steps
        elif not adaptive:
            raise ValueError("Either n_steps or adaptive=True must be set")
        else:
            beta_step = np.nan
        self.adaptive = adaptive
        beta = 0.0

        if min_step is None:
            if max_n_steps is None:
                min_step = 0.0
                self.adaptive_min_step = False
            else:
                min_step = 1 / max_n_steps
                self.adaptive_min_step = True
        else:
            self.adaptive_min_step = False

        iterations = 0
        while True:
            iterations += 1

            beta, min_step = self.determine_beta(
                samples,
                beta,
                beta_step,
                min_step,
            )
            self.history.eff_target.append(
                self.current_target_efficiency(beta)
            )

            logger.info(f"it {iterations} - beta: {beta}")
            self.history.beta.append(beta)

            ess = effective_sample_size(samples.log_weights(beta))
            eff = ess / len(samples)
            if eff < 0.1:
                logger.warning(
                    f"it {iterations} - Low sample efficiency: {eff:.2f}"
                )
            self.history.ess.append(ess)
            logger.info(
                f"it {iterations} - ESS: {ess:.1f} ({eff:.2f} efficiency)"
            )
            self.history.ess_target.append(
                effective_sample_size(samples.log_weights(1.0))
            )

            log_evidence_ratio = samples.log_evidence_ratio(beta)
            log_evidence_ratio_var = samples.log_evidence_ratio_variance(beta)
            self.history.log_norm_ratio.append(log_evidence_ratio)
            self.history.log_norm_ratio_var.append(log_evidence_ratio_var)
            logger.info(
                f"it {iterations} - Log evidence ratio: {log_evidence_ratio:.2f} +/- {np.sqrt(log_evidence_ratio_var):.2f}"
            )

            samples = samples.resample(beta, rng=self.rng)

            samples = self.mutate(samples, beta)
            if beta == 1.0 or (
                max_n_steps is not None and iterations >= max_n_steps
            ):
                break

        # If n_final_samples is not None, perform additional mutations steps
        if n_final_samples is not None:
            logger.info(f"Generating {n_final_samples} final samples")
            final_samples = samples.resample(
                1.0, n_samples=n_final_samples, rng=self.rng
            )
            samples = self.mutate(final_samples, 1.0, n_steps=n_final_steps)

        samples.log_evidence = samples.xp.sum(
            asarray(self.history.log_norm_ratio, self.xp)
        )
        samples.log_evidence_error = samples.xp.sqrt(
            samples.xp.sum(asarray(self.history.log_norm_ratio_var, self.xp))
        )

        final_samples = samples.to_standard_samples()
        logger.info(
            f"Log evidence: {final_samples.log_evidence:.2f} +/- {final_samples.log_evidence_error:.2f}"
        )
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
