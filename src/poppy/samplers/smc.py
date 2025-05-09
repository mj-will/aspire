import logging
from dataclasses import dataclass
from typing import Callable
from functools import partial

import emcee
import numpy as np
from array_api_compat.common._typing import Array

from ..flows.base import Flow
from ..history import SMCHistory
from ..samples import BaseSamples, Samples
from ..utils import effective_sample_size, logsumexp, to_numpy, track_calls
from .base import Sampler

logger = logging.getLogger(__name__)


@dataclass
class SMCSamples(BaseSamples):
    beta: float | None = None
    log_evidence: float | None = None
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

    def resample(self, beta, n_samples: int | None = None) -> "SMCSamples":
        if beta == self.beta:
            logger.warning("Resampling with the same beta value")
            return self
        if n_samples is None:
            n_samples = len(self.x)
        log_w = self.log_weights(beta)
        w = self.xp.exp(log_w - logsumexp(log_w))
        idx = np.random.choice(
            len(self.x), size=n_samples, replace=True, p=w
        )
        return self.__class__(
            x=self.x[idx],
            log_likelihood=self.log_likelihood[idx],
            log_prior=self.log_prior[idx],
            log_q=self.log_q[idx],
            beta=beta,
        )

    def __str__(self):
        out = super().__str__()
        if self.log_evidence is not None:
            out += (
                f"Log evidence: {self.log_evidence:.2f}\n"
            )
        return out

    def to_standard_samples(self):
        """Convert the samples to standard samples."""
        return Samples(
            x=self.x,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            xp=self.xp,
            parameters=self.parameters,
            log_evidence=self.log_evidence,
            log_evidence_error=self.log_evidence_error,
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
                logger.info(
                    f"Final number of samples: {n_final_samples}"
                )
                samples = samples.resample(beta, n_samples=n_final_samples)
            else:
                samples = samples.resample(beta)

            samples = self.mutate(samples, beta)
            if beta == 1.0:
                break

        samples.log_evidence = samples.xp.sum(self.history.log_norm_ratio)
        samples.log_evidence_error = samples.xp.nan
        final_samples = samples.to_standard_samples()
        logger.info(
            f"Log evidence: {final_samples.log_evidence:.2f}"
        )
        return final_samples


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

    def config_dict(self, include_sample_calls = True):
        config =  super().config_dict(include_sample_calls)
        config["preconditioner_kwargs"] = self.pflow_kwargs
        return config
    
class EmceeSMC(SMCSampler):

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = 5,
        adaptive: bool = False,
        target_efficiency: float = 0.5,
        sampler_kwargs: dict | None = None,
        n_final_samples: int | None = None,
    ):
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_kwargs.setdefault("nsteps", 5 * self.dims )
        self.sampler_kwargs.setdefault("progress", True)
        self.emcee_moves = self.sampler_kwargs.pop("moves", None)
        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            n_final_samples=n_final_samples,
        )

    def mutate(self, particles, beta):
        logger.info("Mutating particles")
        sampler = emcee.EnsembleSampler(
            len(particles.x),
            self.dims,
            self.log_prob,
            args=(beta,),
            vectorize=True,
            moves=self.emcee_moves,
        )
        sampler.run_mcmc(to_numpy(particles.x), **self.sampler_kwargs)
        self.history.mcmc_acceptance.append(np.mean(sampler.acceptance_fraction))
        self.history.mcmc_autocorr.append(
            sampler.get_autocorr_time(quiet=True, discard=int(0.2 * self.sampler_kwargs["nsteps"]))
        )
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
    

class EmceePSMC(PreconditionedSMC, EmceeSMC):

    def mutate(self, particles, beta):
        self.train_preconditioner(particles)
        logger.info("Mutating particles")
        sampler = emcee.EnsembleSampler(
            len(particles.x),
            self.dims,
            self.log_prob,
            args=(beta,),
            vectorize=True,
            moves=self.emcee_moves,
        )
        z = to_numpy(self.pflow.forward(particles.x)[0])
        sampler.run_mcmc(z, **self.sampler_kwargs)
        self.history.mcmc_acceptance.append(np.mean(sampler.acceptance_fraction))
        self.history.mcmc_autocorr.append(
            sampler.get_autocorr_time(quiet=True, discard=int(0.2 * self.sampler_kwargs["nsteps"]))
        )
        z = sampler.get_chain(flat=False)[-1, ...]
        x, _ = self.pflow.inverse(z)
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


class MiniCrankSMC(SMCSampler):
    """MiniCrank SMC sampler."""

    rng = None

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = 5,
        adaptive: bool = False,
        target_efficiency: float = 0.5,
        n_final_samples: int | None = None,
        sampler_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_kwargs.setdefault("n_steps", 5 * self.dims)
        self.sampler_kwargs.setdefault("target_acceptance_rate", 0.234)
        self.rng = rng or np.random.default_rng()
        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            n_final_samples=n_final_samples,
        )

    def mutate(self, particles, beta):
        from minicrank import Sampler
        from minicrank.step import TPCNStep

        log_prob_fn = partial(self.log_prob, beta=beta)

        sampler = Sampler(
            log_prob_fn=log_prob_fn,
            step_fn=TPCNStep(self.dims, rng=self.rng),
            rng=self.rng,
            dims=self.dims,
            target_acceptance_rate=self.sampler_kwargs["target_acceptance_rate"],
        )
        chain, history = sampler.sample(
            to_numpy(particles.x),
            n_steps=self.sampler_kwargs["n_steps"],
        )
        x = chain[-1]

        self.history.mcmc_acceptance.append(np.mean(history.acceptance_rate))

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
