from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from array_api_compat import array_namespace, device
from sequince import make_standard_bayesian_smc
from sequince.posterior import draw_additional_samples
from sequince.utils.random import resolve_rng

from ..history import SMCHistory
from ..samples import SMCSamples
from ..utils import asarray, track_calls
from .base import (
    CheckpointCallback,
    CheckpointInput,
    Sampler,
    SamplerCallback,
    load_checkpoint,
)

if TYPE_CHECKING:
    from sequince.posterior import TemperedPosterior
    from sequince.result import SMCResult
    from sequince.state import ParticleState
    from sequince.steps.context import StepContext
    from sequince.steps.weighting import TargetESSRatio
    from sequince.utils.random import RNGLike

logger = logging.getLogger(__name__)


class SMCSampler(Sampler):
    """Sequential Monte Carlo sampler using Sequince's Bayesian pipeline.

    Parameters
    ----------
    prior_flow : NormalizingFlow or FlowProposal
        Fitted proposal in physical coordinates, providing samples of shape
        ``(n_particles, dims)`` and their log densities.
    log_likelihood : callable
        Function mapping a ``BaseSamples`` container to an array of
        per-particle log-likelihood values.
    log_prior : callable
        Function mapping a ``BaseSamples`` container to an array of
        per-particle log-prior values.
    dims : int
        Number of parameter dimensions.
    xp : module
        Array API namespace used for sampling.
    dtype : dtype, str or None, optional
        Sample dtype. If None, use the backend's default dtype.
    parameters : list[str] or None, optional
        Parameter names. Sample containers generate names when omitted.
    preconditioning_transform : ParticleTransform or None, optional
        Transform used for mutation. None selects an identity transform.

    Notes
    -----
    ``result`` retains the native Sequince result. ``history`` exposes
    Aspire's SMC diagnostics and, when requested, particle populations.
    """

    result: SMCResult

    def _state_to_samples(self, state: ParticleState) -> SMCSamples:
        beta = float(state.metadata.get("beta", 0.0))
        particles = state.particles
        samples = SMCSamples(
            x=particles,
            parameters=self.parameters,
            xp=array_namespace(particles),
            dtype=self.dtype,
            beta=beta,
        )
        components = (
            state.evaluation.components if state.evaluation is not None else {}
        )
        log_q = components.get("log_proposal")
        log_prior = components.get("log_prior")
        log_likelihood = components.get("log_likelihood")
        samples.log_q = samples.array_to_namespace(
            self._proposal().log_prob(samples.x) if log_q is None else log_q
        )
        samples.log_prior = samples.array_to_namespace(
            self.log_prior(samples) if log_prior is None else log_prior
        )
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
            if log_likelihood is None
            else log_likelihood
        )
        return samples

    def _sample_history(self, result: SMCResult) -> list[SMCSamples]:
        states = [
            state
            for state in (result.initial_state, *result.states)
            if state is not None
        ]
        return [self._state_to_samples(state).to_numpy() for state in states]

    @staticmethod
    def _mutation_kwargs(
        mutation: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize legacy Aspire mutation options for Sequince."""
        normalized = dict(kwargs)
        if mutation != "blackjax":
            return normalized

        algorithm_kwargs = dict(normalized.pop("algorithm_kwargs", {}) or {})
        for key in (
            "step_size",
            "inverse_mass_matrix",
            "sigma",
            "num_integration_steps",
        ):
            if key in normalized:
                algorithm_kwargs[key] = normalized.pop(key)
        if algorithm_kwargs:
            normalized["algorithm_kwargs"] = algorithm_kwargs
        return normalized

    def _posterior_samples(
        self,
        result: SMCResult,
        n_samples: int,
        rng: RNGLike,
        *,
        posterior: TemperedPosterior,
        mutation_n_steps: int,
        mutation_kwargs: dict[str, Any],
    ) -> SMCSamples:
        """Draw final samples while retaining cached target values."""
        state = result.final_state
        rng_obj = resolve_rng(rng, xp=self.xp)
        if n_samples > len(state.particles):
            state, _ = draw_additional_samples(
                result=result,
                rng=rng_obj,
                n_particles=n_samples,
                mutation=self.mutation,
                mutation_n_steps=mutation_n_steps,
                mutation_kwargs=mutation_kwargs,
                particle_transform=self._particle_transform(),
                posterior=posterior,
            )
            return self._state_to_samples(state)

        indices = rng_obj.choice(
            len(state.particles),
            size=n_samples,
            replace=True,
            p=state.weights,
        )
        xp = array_namespace(state.particles)
        indices = asarray(
            indices,
            xp=xp,
            device=device(state.particles),
        )
        return self._state_to_samples(state.resample(indices))

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int | None = None,
        adaptive: bool = True,
        min_beta_step: float | None = None,
        max_beta_step: float | None = None,
        max_n_steps: int | None = None,
        target_efficiency: float | tuple[float, float] = 0.5,
        target_efficiency_rate: float = 1.0,
        n_final_samples: int | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        rng: RNGLike = None,
        callbacks: Sequence[SamplerCallback] | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | Path | None = None,
        resume_from: CheckpointInput | None = None,
        store_sample_history: bool = True,
        beta_tolerance: float | None = None,
    ) -> SMCSamples:
        """Draw samples using adaptive or fixed-schedule SMC.

        Parameters
        ----------
        n_samples : int
            Number of particles in the SMC population.
        n_steps : int or None, optional
            Number of tempering steps when ``adaptive=False``; required for
            a fixed schedule. Adaptive runs select their own beta schedule.
        adaptive : bool, optional
            Adapt beta increments to the target ESS ratio. Default is True.
        min_beta_step : float or None, optional
            Minimum adaptive beta increment. None uses ``1e-8``. In the
            current Sequince standard sampler, this also sets the beta-search
            tolerance.
        max_beta_step : float or None, optional
            Maximum adaptive beta increment. None imposes no additional cap
            beyond the remaining interval to beta one.
        max_n_steps : int or None, optional
            Maximum number of adaptive SMC iterations. None imposes no
            iteration limit. Reaching the limit may return an intermediate
            target with ``samples.beta < 1``; it does not force beta to one.
        target_efficiency : float or tuple[float, float], optional
            Target ESS ratio, default 0.5. A pair ``(start, end)`` selects
            ``start + (end - start) * beta**target_efficiency_rate``.
            Target ratios must lie in ``(0, 1]``.
        target_efficiency_rate : float, optional
            Exponent for a beta-dependent target efficiency. Default is 1.0.
        n_final_samples : int or None, optional
            Number of returned samples; None uses ``n_samples``. The final
            population is resampled with replacement. Increasing its size
            also applies additional mutation moves.
        sampler_kwargs : dict[str, Any] or None, optional
            Mutation options forwarded to Sequince. ``n_steps`` (or
            ``nsteps``) sets mutation steps per SMC iteration, defaulting to
            ``5 * dims``. ``n_final_steps`` controls additional mutation when
            increasing the final sample count. Remaining options configure
            the selected mutation backend.
        rng : RandomGenerator, numpy.random.Generator, int or None, optional
            Random generator or seed. When resuming, the checkpoint's saved
            random stream is used, including for final posterior resampling.
        callbacks : sequence of callable or None, optional
            Native Sequince iteration callbacks receiving
            ``(state, context, result)``. A callback may return an updated
            context or None.
        checkpoint_callback : callable or None, optional
            Persistence callback receiving a committed ``SMCCheckpoint`` at
            the checkpoint cadence. May be used alongside file checkpointing.
        checkpoint_every : int or None, optional
            Number of sampler iterations between checkpoints. None uses one
            when a checkpoint file or callback is configured.
        checkpoint_file_path : str, pathlib.Path or None, optional
            Checkpoint destination. ``.h5``/``.hdf5`` selects HDF5;
            ``.pkl``/``.pickle`` selects Sequince's native pickle format.
            None disables file checkpointing.
        resume_from : SMCCheckpoint, mapping, str, pathlib.Path or None, optional
            Native checkpoint, its dictionary representation, or an HDF5 or
            pickle checkpoint path. None starts a new run. Pre-migration
            Aspire checkpoints are incompatible.
        store_sample_history : bool, optional
            Store particle populations in ``history.sample_history``.
            Default is True. Diagnostic histories are retained when False.
        beta_tolerance : float or None, optional
            Legacy compatibility argument. Supplying a value emits a
            ``DeprecationWarning``; it currently has no effect with this
            adapter. Default is None.

        Returns
        -------
        SMCSamples
            Resampled particles of shape ``(n_final_samples, dims)`` with
            target-density components, beta, log evidence, and its estimated
            uncertainty. If stopped early, these describe the intermediate
            tempered target rather than the full posterior.

        Raises
        ------
        ValueError
            If the schedule, proposal shape, target ESS ratio, or target
            values are invalid. NaN and positive-infinite target values are
            rejected; negative infinity represents zero target density.
        """
        if beta_tolerance is not None:
            warnings.warn(
                "beta_tolerance is deprecated and has no effect with the "
                "Sequince backend. Remove this argument; min_beta_step and "
                "max_beta_step control beta increments, not search tolerance.",
                DeprecationWarning,
                stacklevel=3,
            )
        rng = resolve_rng(rng, xp=self.xp)
        mutation_kwargs = dict(sampler_kwargs or {})
        configured_final_steps = mutation_kwargs.pop("n_final_steps", None)
        mutation_n_steps = int(
            mutation_kwargs.pop(
                "n_steps", mutation_kwargs.pop("nsteps", 5 * self.dims)
            )
        )
        final_mutation_n_steps = (
            mutation_n_steps
            if configured_final_steps is None
            else int(configured_final_steps)
        )
        progress = mutation_kwargs.pop("progress", None)
        if progress is not None and self.mutation == "emcee":
            mutation_kwargs.setdefault("run_kwargs", {})["progress"] = progress
        mutation_kwargs = self._mutation_kwargs(self.mutation, mutation_kwargs)

        target_ess_ratio: TargetESSRatio
        if isinstance(target_efficiency, tuple):
            start, end = map(float, target_efficiency)

            def evolving_target_ess_ratio(
                temperature: float, context: StepContext | None = None
            ) -> float:
                return start + (end - start) * (
                    temperature**target_efficiency_rate
                )

            target_ess_ratio = evolving_target_ess_ratio
        else:
            target_ess_ratio = float(target_efficiency)

        def history_target_efficiency(temperature: float) -> float:
            if callable(target_ess_ratio):
                return target_ess_ratio(temperature)
            return target_ess_ratio

        smc, posterior, metadata = make_standard_bayesian_smc(
            self._log_prior_array,
            self._log_likelihood_array,
            log_proposal=self._proposal(),
            target_ess_ratio=target_ess_ratio,
            min_delta=min_beta_step or 1e-8,
            max_delta=max_beta_step,
            adaptive=adaptive,
            num_steps=n_steps,
            mutation=self.mutation,
            mutation_n_steps=mutation_n_steps,
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
            result = smc.run(
                self._initial_state(n_samples, posterior, rng=rng),
                num_steps=max_n_steps if adaptive else n_steps,
                rng=rng,
                metadata=metadata,
            )
        else:
            runtime = smc.resume_from(load_checkpoint(resume_from))
            # Final posterior resampling must continue the restored stream too.
            rng = runtime.rng
            result = smc.run(
                runtime=runtime,
                num_steps=max_n_steps if adaptive else n_steps,
            )

        self.result = result
        sample_history = (
            self._sample_history(result) if store_sample_history else []
        )
        self.history = SMCHistory.from_sequince_result(
            result,
            sample_history=sample_history,
            target_efficiency=history_target_efficiency,
        )
        samples = self._posterior_samples(
            result,
            n_final_samples or n_samples,
            rng,
            posterior=posterior,
            mutation_n_steps=final_mutation_n_steps,
            mutation_kwargs=mutation_kwargs,
        )
        samples.beta = float(result.context.metadata["beta"])
        if samples.beta < 1.0:
            logger.warning(
                "SMC stopped at beta=%s before reaching the posterior. "
                "Returned samples follow the intermediate tempered target; "
                "resume the checkpoint or increase max_n_steps.",
                samples.beta,
            )
        samples.log_evidence = float(result.final_state.log_evidence)
        samples.log_evidence_error = math.sqrt(
            sum(self.history.log_norm_ratio_var)
        )
        return samples


class MiniPCNSMC(SMCSampler):
    """Sequince SMC with MiniPCN mutation."""


class EmceeSMC(SMCSampler):
    """Sequince SMC with emcee mutation."""

    mutation = "emcee"


class BlackJAXSMC(SMCSampler):
    """Sequince SMC with BlackJAX mutation."""

    mutation = "blackjax"


__all__ = ["BlackJAXSMC", "EmceeSMC", "MiniPCNSMC", "SMCSampler"]
