import numpy as np

from ..samples import Samples, to_numpy
from ..utils import track_calls
from .base import Sampler


class MCMCSampler(Sampler):
    def draw_initial_samples(self, n_samples: int) -> Samples:
        """Draw initial samples from the prior flow."""
        # Flow may propose samples outside prior bounds, so we may need
        # to try multiple times to get enough valid samples.
        n_samples_drawn = 0
        samples = None
        while n_samples_drawn < n_samples:
            n_to_draw = n_samples - n_samples_drawn
            x, log_q = self.prior_flow.sample_and_log_prob(n_to_draw)
            new_samples = Samples(x, xp=self.xp, log_q=log_q)
            new_samples.log_prior = new_samples.array_to_namespace(
                self.log_prior(new_samples)
            )
            valid = self.xp.isfinite(new_samples.log_prior)
            n_valid = self.xp.sum(valid)
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
        samples = Samples(x, xp=self.xp)
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

        rng = rng or np.random.default_rng()

        samples = self.draw_initial_samples(nwalkers)
        p0 = samples.x

        z0 = to_numpy(self.preconditioning_transform.fit(p0))

        self.sampler.run_mcmc(z0, nsteps, **kwargs)

        z = self.sampler.get_chain(flat=True, discard=discard)
        x = self.preconditioning_transform.inverse(z)[0]

        x_evidence, log_q = self.prior_flow.sample_and_log_prob(n_samples)
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

        rng = rng or np.random.default_rng()
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
        else:
            z = chain[burnin::thin].reshape(-1, self.dims)

        x = self.preconditioning_transform.inverse(z)[0]

        samples_mcmc = Samples(x, xp=self.xp, parameters=self.parameters)
        samples_mcmc.log_prior = samples_mcmc.array_to_namespace(
            self.log_prior(samples_mcmc)
        )
        samples_mcmc.log_likelihood = samples_mcmc.array_to_namespace(
            self.log_likelihood(samples_mcmc)
        )
        return samples_mcmc
