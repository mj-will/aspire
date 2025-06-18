from functools import partial

import numpy as np

from ...samples import SMCSamples
from ...utils import to_numpy, track_calls
from .base import NumpySMCSampler


class MiniPCNSMC(NumpySMCSampler):
    """MiniPCN SMC sampler."""

    rng = None

    def log_prob(self, x, beta=None):
        return to_numpy(super().log_prob(x, beta))

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = 5,
        adaptive: bool = False,
        target_efficiency: float = 0.5,
        n_final_samples: int | None = None,
        minipcn_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.minipcn_kwargs = minipcn_kwargs or {}
        self.minipcn_kwargs.setdefault("n_steps", 5 * self.dims)
        self.minipcn_kwargs.setdefault("target_acceptance_rate", 0.234)
        self.rng = rng or np.random.default_rng()
        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            n_final_samples=n_final_samples,
        )

    def mutate(self, particles, beta):
        from minipcn import Sampler
        from minipcn.step import TPCNStep

        log_prob_fn = partial(self.log_prob, beta=beta)

        sampler = Sampler(
            log_prob_fn=log_prob_fn,
            step_fn=TPCNStep(self.dims, rng=self.rng),
            rng=self.rng,
            dims=self.dims,
            target_acceptance_rate=self.minipcn_kwargs[
                "target_acceptance_rate"
            ],
        )
        # Map to transformed dimension for sampling
        z = to_numpy(self.fit_preconditioning_transform(particles.x))
        chain, history = sampler.sample(
            z,
            n_steps=self.minipcn_kwargs["n_steps"],
        )
        x = self.preconditioning_transform.inverse(chain[-1])[0]

        self.history.mcmc_acceptance.append(np.mean(history.acceptance_rate))

        samples = SMCSamples(x, xp=self.xp, beta=beta)
        samples.log_q = samples.array_to_namespace(
            self.prior_flow.log_prob(samples.x)
        )
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        if np.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")
        return samples
