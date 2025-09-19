import copy
import logging

import numpy as np

from ...samples import SMCSamples
from ...utils import track_calls
from .base import NumpySMCSampler

logger = logging.getLogger(__name__)


class EmceeSMC(NumpySMCSampler):
    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = None,
        adaptive: bool = True,
        target_efficiency: float = 0.5,
        target_efficiency_rate: float = 1.0,
        sampler_kwargs: dict | None = None,
        n_final_samples: int | None = None,
    ):
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_kwargs.setdefault("nsteps", 5 * self.dims)
        self.sampler_kwargs.setdefault("progress", True)
        self.emcee_moves = self.sampler_kwargs.pop("moves", None)
        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            target_efficiency_rate=target_efficiency_rate,
            n_final_samples=n_final_samples,
        )

    def mutate(self, particles, beta, n_steps=None):
        import emcee

        logger.info("Mutating particles")
        sampler = emcee.EnsembleSampler(
            len(particles.x),
            self.dims,
            self.log_prob,
            args=(beta,),
            vectorize=True,
            moves=self.emcee_moves,
        )
        z = self.fit_preconditioning_transform(particles.x)
        kwargs = copy.deepcopy(self.sampler_kwargs)
        if n_steps is not None:
            kwargs["nsteps"] = n_steps
        sampler.run_mcmc(z, **kwargs)
        self.history.mcmc_acceptance.append(
            np.mean(sampler.acceptance_fraction)
        )
        self.history.mcmc_autocorr.append(
            sampler.get_autocorr_time(
                quiet=True, discard=int(0.2 * self.sampler_kwargs["nsteps"])
            )
        )
        z = sampler.get_chain(flat=False)[-1, ...]
        x = self.preconditioning_transform.inverse(z)[0]
        samples = SMCSamples(x, xp=self.xp, beta=beta)
        samples.log_q = samples.array_to_namespace(
            self.prior_flow.log_prob(samples.x)
        )
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        if samples.xp.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")
        return samples
