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
    """Bayesian SMC whose execution and checkpointing are owned by Sequince."""

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
        """Sample with SMC; ``beta_tolerance`` is deprecated and ignored."""
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
