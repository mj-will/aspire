from __future__ import annotations

from array_api_compat import array_namespace
from sequince import make_mcmc
from sequince.utils.random import resolve_rng

from ..samples import MCMCSamples, Samples
from ..utils import track_calls
from .base import Sampler, load_checkpoint


class MCMCSampler(Sampler):
    """Standalone MCMC whose execution and checkpointing are owned by Sequince."""

    @track_calls
    def sample(
        self,
        n_samples: int | None = None,
        n_walkers: int | None = None,
        nwalkers: int | None = None,
        n_steps: int = 100,
        nsteps: int | None = None,
        thin: int = 1,
        burnin: int = 0,
        discard: int | None = None,
        last_step_only: bool = False,
        rng=None,
        callbacks=None,
        checkpoint_callback=None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | None = None,
        resume_from=None,
        **mutation_kwargs,
    ):
        if nsteps is not None:
            n_steps = nsteps
        if discard is not None:
            burnin = discard
        walkers = n_walkers or nwalkers or n_samples
        if walkers is None:
            raise ValueError("n_samples or a walker count must be provided.")

        rng = resolve_rng(rng, xp=self.xp)

        progress = mutation_kwargs.pop("progress", None)
        if progress is not None and self.mutation == "emcee":
            mutation_kwargs.setdefault("run_kwargs", {})["progress"] = progress

        mcmc = make_mcmc(
            self._log_posterior_array,
            mutation=self.mutation,
            n_steps=n_steps,
            mutation_kwargs=mutation_kwargs,
            particle_transform=self._particle_transform(),
            callbacks=callbacks,
            checkpoint_writers=self._checkpoint_writers(
                checkpoint_file_path,
                checkpoint_every,
                checkpoint_callback,
            ),
        )
        if resume_from is None:
            result = mcmc.run(
                self._mcmc_initial_state(walkers, rng=rng),
                rng=rng,
            )
        else:
            result = mcmc.run(
                runtime=mcmc.resume_from(load_checkpoint(resume_from))
            )

        self.result = result
        self.history = result
        chain = result.samples(burn_in=burnin, thin=thin, flat=False)
        if last_step_only:
            particles = chain[-1]
            samples = Samples(
                x=particles,
                parameters=self.parameters,
                xp=array_namespace(particles),
                dtype=self.dtype,
            )
            samples.log_prior = samples.array_to_namespace(
                self.log_prior(samples)
            )
            samples.log_likelihood = samples.array_to_namespace(
                self.log_likelihood(samples)
            )
            return samples

        samples = MCMCSamples.from_chain(
            chain,
            parameters=self.parameters,
            xp=array_namespace(chain),
            dtype=self.dtype,
            thin=thin,
            burn_in=burnin,
        )
        return samples if n_samples is None else samples[:n_samples]


class MiniPCN(MCMCSampler):
    """Sequince-backed MiniPCN MCMC."""


class Emcee(MCMCSampler):
    """Sequince-backed emcee MCMC."""

    mutation = "emcee"


__all__ = ["Emcee", "MCMCSampler", "MiniPCN"]
