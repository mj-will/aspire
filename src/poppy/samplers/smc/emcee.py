import logging

from .base import SMCSampler, PreconditionedSMC
from ...samples import SMCSamples
from ...utils import to_numpy, track_calls

logger = logging.getLogger(__name__)


class EmceeSMC(SMCSampler):

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = 5,
        adaptive: bool = False,
        target_efficiency: float = 0.5,
        emcee_kwargs: dict | None = None,
        n_final_samples: int | None = None,
    ):
        self.emcee_kwargs = emcee_kwargs or {}
        self.emcee_kwargs.setdefault("nsteps", 5 * self.dims )
        self.emcee_kwargs.setdefault("progress", True)
        self.emcee_moves = self.emcee_kwargs.pop("moves", None)
        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            n_final_samples=n_final_samples,
        )

    def mutate(self, particles, beta):
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
        sampler.run_mcmc(to_numpy(particles.x), **self.emcee_kwargs)
        self.history.mcmc_acceptance.append(np.mean(sampler.acceptance_fraction))
        self.history.mcmc_autocorr.append(
            sampler.get_autocorr_time(quiet=True, discard=int(0.2 * self.emcee_kwargs["nsteps"]))
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
        import emcee

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
        sampler.run_mcmc(z, **self.emcee_kwargs)
        self.history.mcmc_acceptance.append(np.mean(sampler.acceptance_fraction))
        self.history.mcmc_autocorr.append(
            sampler.get_autocorr_time(quiet=True, discard=int(0.2 * self.emcee_kwargs["nsteps"]))
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
