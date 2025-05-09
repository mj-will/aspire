import numpy as np

from ..samples import Samples, to_numpy
from ..utils import track_calls
from .base import Sampler


class MCMCSampler(Sampler):
    pass


class Emcee(MCMCSampler):
    def log_prob(self, z):
        x, log_abs_det_jacobian = self.flow.inverse(z)
        samples = Samples(x, xp=self.xp)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = (
            samples.log_likelihood
            + samples.log_prior
            + samples.array_to_namespace(log_abs_det_jacobian)
        )
        return to_numpy(log_prob).flatten()

    @track_calls
    def sample(
        self,
        n_samples: int,
        nwalkers: int = None,
        nsteps: int = 500,
        rng=None,
        discard=0,
        **kwargs,
    ) -> Samples:
        from emcee import EnsembleSampler

        nwalkers = nwalkers or n_samples
        self.sampler = EnsembleSampler(
            nwalkers,
            self.dims,
            log_prob_fn=self.log_prob,
            vectorize=True,
        )

        rng = rng or np.random.default_rng()
        p0 = rng.standard_normal((nwalkers, self.dims))
        self.sampler.run_mcmc(p0, nsteps, **kwargs)

        z = self.sampler.get_chain(flat=True, discard=discard)
        x = self.flow.inverse(z)[0]

        x_evidence, log_q = self.flow.sample_and_log_prob(n_samples)
        samples_evidence = Samples(x_evidence, log_q=log_q, xp=self.xp)
        samples_evidence.log_prior = self.log_prior(samples_evidence)
        samples_evidence.log_likelihood = self.log_likelihood(samples_evidence)
        samples_evidence.compute_weights()

        samples_mcmc = Samples(x, xp=self.xp, parameters=self.parameters)
        samples_mcmc.log_prior = samples_mcmc.array_to_namespace(
            self.log_prior(samples_mcmc)
        )
        samples_mcmc.log_likelihood = samples_mcmc.array_to_namespace(
            self.log_likelihood(samples_mcmc)
        )
        samples_mcmc.log_evidence = samples_mcmc.array_to_namespace(
            samples_evidence.log_evidence
        )
        samples_mcmc.log_evidence_error = samples_mcmc.array_to_namespace(
            samples_evidence.log_evidence_error
        )

        return samples_mcmc
    

class MiniCrank(MCMCSampler):

    def log_prob(self, z):
        x, log_abs_det_jacobian = self.flow.inverse(z)
        samples = Samples(x, xp=self.xp)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = (
            samples.log_likelihood
            + samples.log_prior
            + samples.array_to_namespace(log_abs_det_jacobian)
        )
        return to_numpy(log_prob).flatten()
    
    @track_calls
    def sample(
        self,
        n_samples,
        rng=None,
        target_acceptance_rate=0.234,
        n_steps=100,
        thin=1,
        burnin=0,
        last_step_only=False,
    ):
        from minicrank import Sampler
        from minicrank.step import TPCNStep

        rng = rng or np.random.default_rng()
        x_init = rng.standard_normal((n_samples, self.dims))

        self.sampler = Sampler(
            log_prob_fn=self.log_prob,
            step_fn=TPCNStep(self.dims, rng=rng),
            rng=rng,
            dims=self.dims,
            target_acceptance_rate=target_acceptance_rate,
        )
        
        chain, history = self.sampler.sample(x_init, n_steps=n_steps)

        if last_step_only:
            z = chain[-1]
        else:
            z = chain[burnin::thin].reshape(-1, self.dims)

        x = self.flow.inverse(z)[0] 

        samples_mcmc = Samples(x, xp=self.xp, parameters=self.parameters)
        samples_mcmc.log_prior = samples_mcmc.array_to_namespace(
            self.log_prior(samples_mcmc)
        )
        samples_mcmc.log_likelihood = samples_mcmc.array_to_namespace(
            self.log_likelihood(samples_mcmc)
        )
        return samples_mcmc