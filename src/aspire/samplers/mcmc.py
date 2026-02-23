import numpy as np

from ..samples import MCMCSamples, Samples, to_numpy
from ..utils import track_calls
from .base import Sampler


class MCMCSampler(Sampler):
    def __init__(
        self,
        log_likelihood,
        log_prior,
        dims,
        prior_flow,
        xp,
        dtype=None,
        parameters=None,
        preconditioning_transform=None,
        rng=None,
    ):
        super().__init__(
            log_likelihood,
            log_prior,
            dims,
            prior_flow,
            xp,
            dtype,
            parameters,
            preconditioning_transform,
        )
        self.rng = rng or np.random.default_rng()

    def draw_initial_samples(self, n_samples: int) -> Samples:
        """Draw initial samples from the prior flow."""
        # Flow may propose samples outside prior bounds, so we may need
        # to try multiple times to get enough valid samples.
        n_samples_drawn = 0
        samples = None
        while n_samples_drawn < n_samples:
            x, log_q = self.prior_flow.sample_and_log_prob(n_samples)
            new_samples = Samples(
                x,
                xp=self.xp,
                log_q=log_q,
                dtype=self.dtype,
                parameters=self.parameters,
            )
            new_samples.log_prior = new_samples.array_to_namespace(
                self.log_prior(new_samples)
            )
            valid = self.xp.isfinite(new_samples.log_prior)
            n_valid = int(self.xp.sum(valid))
            if n_valid > 0:
                if samples is None:
                    samples = new_samples[valid]
                else:
                    samples = Samples.concatenate(
                        [samples, new_samples[valid]]
                    )
                n_samples_drawn += n_valid

        if n_samples_drawn > n_samples:
            samples = samples[:n_samples]

        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        return samples

    def log_prob(self, z):
        """Compute the log probability of the samples.

        Input samples are in the transformed space.
        """
        x, log_abs_det_jacobian = self.preconditioning_transform.inverse(z)
        samples = Samples(x, xp=self.xp, dtype=self.dtype)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = (
            samples.log_likelihood
            + samples.log_prior
            + samples.array_to_namespace(log_abs_det_jacobian)
        )
        return to_numpy(log_prob).flatten()


class Emcee(MCMCSampler):
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

        rng = rng or self.rng or np.random.default_rng()

        samples = self.draw_initial_samples(nwalkers)
        p0 = samples.x

        z0 = to_numpy(self.preconditioning_transform.fit(p0))

        self.sampler.run_mcmc(z0, nsteps, **kwargs)

        chain = self.sampler.get_chain(discard=discard)

        # Transform chain back to original space
        chain_z = chain.reshape(-1, self.dims)
        chain_x, log_jacobian = self.preconditioning_transform.inverse(chain_z)
        chain_x = chain_x.reshape(chain.shape)
        # Create MCMCSamples
        samples_mcmc = MCMCSamples.from_chain(
            chain=chain_x,
            parameters=self.parameters,
            xp=self.xp,
            dtype=self.dtype,
            burn_in=discard,
        )

        return samples_mcmc


class MiniPCN(MCMCSampler):
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
        step_fn="tpcn",
    ):
        from minipcn import Sampler

        rng = rng or self.rng or np.random.default_rng()
        p0 = self.draw_initial_samples(n_samples).x

        z0 = to_numpy(self.preconditioning_transform.fit(p0))

        self.sampler = Sampler(
            log_prob_fn=self.log_prob,
            step_fn=step_fn,
            rng=rng,
            dims=self.dims,
            target_acceptance_rate=target_acceptance_rate,
        )

        chain, history = self.sampler.sample(z0, n_steps=n_steps)

        if last_step_only:
            z = chain[-1]
            x = self.preconditioning_transform.inverse(z)[0]
            samples_mcmc = Samples(x, xp=self.xp, parameters=self.parameters)
            samples_mcmc.log_prior = samples_mcmc.array_to_namespace(
                self.log_prior(samples_mcmc)
            )
            samples_mcmc.log_likelihood = samples_mcmc.array_to_namespace(
                self.log_likelihood(samples_mcmc)
            )
        else:
            # Apply burn-in and thinning
            chain_thinned = chain[burnin::thin]

            # Transform chain back to original space
            chain_z = chain_thinned.reshape(-1, self.dims)
            chain_x, _ = self.preconditioning_transform.inverse(chain_z)
            chain_x = chain_x.reshape(chain_thinned.shape)

            # Create MCMCSamples
            samples_mcmc = MCMCSamples.from_chain(
                chain=chain_x,
                parameters=self.parameters,
                xp=self.xp,
                dtype=self.dtype,
                thin=thin,
                burn_in=burnin,
            )

        return samples_mcmc


class ParallelTemperedMCMCSampler(MCMCSampler):
    """Wrapper for Parallel Tempered MCMC Samplers"""

    def log_likelihood_wrapper(self, z):
        """Wrapper for log-likelihood that takes array inputs."""
        x, log_abs_det_jacobian = self.preconditioning_transform.inverse(z)
        samples = Samples(x, xp=self.xp, dtype=self.dtype)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        return to_numpy(samples.log_likelihood + log_abs_det_jacobian)

    def log_prior_wrapper(self, z):
        """Wrapper for log-prior that takes array inputs."""
        x, _ = self.preconditioning_transform.inverse(z)
        samples = Samples(x, xp=self.xp, dtype=self.dtype)
        # Skip Jacobian to avoid double counting in log_prior and
        # log_likelihood
        return self.log_prior(samples)
