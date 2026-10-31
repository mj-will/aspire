from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from array_api_compat import array_namespace
from sequince import make_mcmc
from sequince.utils.random import resolve_rng

from ..samples import MCMCSamples, Samples
from ..utils import track_calls
from .base import (
    CheckpointCallback,
    CheckpointInput,
    Sampler,
    SamplerCallback,
    load_checkpoint,
)

if TYPE_CHECKING:
    from sequince.mcmc import MCMCResult
    from sequince.utils.random import RNGLike


class MCMCSampler(Sampler):
    """Standalone MCMC whose execution and checkpointing are owned by Sequince."""

    result: MCMCResult

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
        rng: RNGLike = None,
        callbacks: Sequence[SamplerCallback] | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | Path | None = None,
        resume_from: CheckpointInput | None = None,
        **mutation_kwargs: Any,
    ) -> MCMCSamples | Samples:
        """Run or resume a standalone Sequince MCMC chain.

        Parameters
        ----------
        n_samples : int or None, optional
            Maximum number of flattened samples to return. Also supplies the
            walker count if neither ``n_walkers`` nor ``nwalkers`` is set.
            None returns all retained samples and requires a walker count.
            ``last_step_only=True`` overrides the returned sample-count limit.
        n_walkers : int or None, optional
            Number of walkers. Takes precedence over the other count options.
        nwalkers : int or None, optional
            Compatibility alias for the walker count.
        n_steps : int, optional
            Number of new chain steps per walker, default 100. Resuming
            extends the stored chain by this many steps.
        nsteps : int or None, optional
            Compatibility alias overriding ``n_steps`` when supplied.
        thin : int, optional
            Retain every ``thin``-th chain step. Default is one.
        burnin : int, optional
            Number of initial chain steps to discard from the returned
            samples. Default is zero.
        discard : int or None, optional
            Compatibility alias overriding ``burnin`` when supplied.
        last_step_only : bool, optional
            Return the last retained walker population as ``Samples``
            instead of a flattened ``MCMCSamples`` container. Default is False.
        rng : RandomGenerator, numpy.random.Generator, int or None, optional
            Random generator or seed for a new run. A resumed run restores
            its generator from the checkpoint.
        callbacks : sequence of callable or None, optional
            Native Sequince callbacks receiving ``(state, context, result)``
            after each iteration. May return an updated context or None.
        checkpoint_callback : callable or None, optional
            Callback receiving a committed ``SMCCheckpoint`` at the configured
            cadence. Sequince uses the same checkpoint type for SMC and MCMC.
        checkpoint_every : int or None, optional
            Number of iterations between checkpoints. None uses one when
            a checkpoint destination or callback is configured.
        checkpoint_file_path : str, pathlib.Path or None, optional
            HDF5 (``.h5``/``.hdf5``) or pickle (``.pkl``/``.pickle``)
            checkpoint destination. None disables file checkpointing.
        resume_from : SMCCheckpoint, mapping, str, pathlib.Path or None, optional
            Native checkpoint, its dictionary representation, or a checkpoint
            file path. None starts a new chain. Legacy Aspire checkpoints
            cannot be resumed.
        **mutation_kwargs : Any
            Options for the selected Sequince mutation backend. For emcee,
            ``progress`` is forwarded to its run options. Chain-length and
            mutation options are supplied directly to this method.

        Returns
        -------
        MCMCSamples or Samples
            Retained chain samples of shape ``(n_retained, dims)``, or the
            last walker population of shape ``(n_walkers, dims)`` when
            ``last_step_only=True``. The native chain and diagnostics remain
            available through ``result`` and ``history``.

        Raises
        ------
        ValueError
            If no walker count or sample count is supplied, or if the
            checkpoint, proposal, or target values are invalid.
        """
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
        samples: Samples | MCMCSamples
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
