"""Shared sampler state and Sequince protocol adapters."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import h5py
from array_api_compat import array_namespace
from sequince import draw_initial_state, draw_mcmc_initial_state
from sequince.checkpoint import (
    CheckpointWriter,
    HDF5CheckpointWriter,
    SMCCheckpoint,
)
from sequince.proposals import FlowProposal
from sequince.steps.context import StepContext
from sequince.transforms import IdentityTransform
from sequince.typing import Array
from sequince.utils.random import RNGLike

from ..samples import Samples
from ..utils import determine_backend_name

logger = logging.getLogger(__name__)


class _ValidatedFlowProposal(FlowProposal):
    """Check Aspire's event shape before Sequince builds target caches."""

    def __init__(self, flow: Any, dims: int) -> None:
        super().__init__(flow)
        self._dims = dims

    def _validate(self, particles: Array) -> None:
        if particles.ndim != 2 or particles.shape[1] != self._dims:
            raise ValueError(
                "Proposal particles must have shape "
                f"(n_samples, {self._dims}); received {particles.shape}."
            )

    def sample(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RNGLike = None,
        xp: Any = None,
    ) -> Array:
        particles = super().sample(n_samples, context, rng=rng, xp=xp)
        self._validate(particles)
        return particles

    def sample_and_log_prob(
        self,
        n_samples: int,
        context: StepContext | None = None,
        *,
        rng: RNGLike = None,
        xp: Any = None,
    ) -> tuple[Array, Array]:
        particles, log_prob = super().sample_and_log_prob(
            n_samples, context, rng=rng, xp=xp
        )
        self._validate(particles)
        return particles, log_prob


class CallbackCheckpointWriter(CheckpointWriter):
    """Expose Sequince checkpoints through Aspire's callback convention."""

    def __init__(
        self,
        callback: Callable[[SMCCheckpoint], Any],
        *,
        every: int = 1,
    ) -> None:
        super().__init__(every=every)
        self._callback = callback

    def _write(self, checkpoint: SMCCheckpoint) -> None:
        self._callback(checkpoint)


def load_checkpoint(checkpoint):
    """Load a Sequince checkpoint from an object, mapping, bytes, or HDF5 file."""
    if checkpoint is None or isinstance(checkpoint, SMCCheckpoint):
        return checkpoint
    if isinstance(checkpoint, Mapping):
        if "samples" in checkpoint:
            raise ValueError(
                "Legacy Aspire checkpoints are not compatible with the "
                "Sequince backend. Start a new sampling run."
            )
        return SMCCheckpoint.from_dict(dict(checkpoint))
    if isinstance(checkpoint, (bytes, bytearray)):
        loaded = pickle.loads(checkpoint)
        if isinstance(loaded, SMCCheckpoint):
            return loaded
        if isinstance(loaded, Mapping):
            return load_checkpoint(loaded)
        raise TypeError(
            "Checkpoint bytes did not contain a Sequince checkpoint"
        )
    if isinstance(checkpoint, (str, Path)):
        with h5py.File(checkpoint, "r") as handle:
            group = handle.get("checkpoint")
            if (
                group is not None
                and "state" in group
                and "runtime" not in group
                and "runtime.previous" not in group
            ):
                raise ValueError(
                    "Legacy Aspire checkpoints are not compatible with the "
                    "Sequince backend. Start a new sampling run."
                )
            return SMCCheckpoint.load_hdf5(handle)
    raise TypeError(
        f"Unsupported checkpoint type: {type(checkpoint).__name__}"
    )


class Sampler:
    """Common Aspire sampler state and Sequince integration."""

    mutation = "minipcn"

    def __init__(
        self,
        prior_flow,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        xp: Callable,
        dtype: Any | str | None = None,
        parameters: list[str] | None = None,
        preconditioning_transform=None,
    ):
        self.prior_flow = prior_flow
        self.prior_proposal = _ValidatedFlowProposal(
            prior_flow.flow
            if isinstance(prior_flow, FlowProposal)
            else prior_flow,
            dims,
        )
        self._log_likelihood = log_likelihood
        self._log_prior = log_prior
        self.dims = dims
        self.xp = xp
        self.backend_str = determine_backend_name(xp=self.xp)
        self.dtype = dtype
        self.parameters = parameters
        self.history = None
        self.n_likelihood_evaluations = 0
        self.preconditioning_transform = (
            IdentityTransform()
            if preconditioning_transform is None
            else preconditioning_transform
        )

    def log_likelihood(self, samples: Samples):
        """Evaluate the log likelihood and update the evaluation counter."""
        self.n_likelihood_evaluations += len(samples)
        return self._log_likelihood(samples)

    def log_prior(self, samples: Samples):
        """Evaluate the log prior."""
        return self._log_prior(samples)

    def config_dict(
        self,
        include_sample_calls: str | bool = "last",
    ) -> dict:
        """Return sampler identification and optionally tracked sample calls."""
        config = {"sampler_class": self.__class__.__name__}
        if include_sample_calls is False:
            return config
        if include_sample_calls is True:
            include_sample_calls = "all"
        if not isinstance(include_sample_calls, str):
            raise ValueError(
                "include_sample_calls must be 'last', 'all', or False; "
                f"received {include_sample_calls!r}"
            )

        calls = getattr(self.sample, "calls", None)
        if calls is None:
            logger.warning(
                "Sampler does not have a sample method with calls attribute."
            )
            return config
        if include_sample_calls.lower() == "last":
            call_kwargs = dict(calls.kwargs[-1]) if calls.kwargs else None
            if call_kwargs is not None:
                for runtime_key in (
                    "callbacks",
                    "checkpoint_callback",
                    "resume_from",
                ):
                    call_kwargs.pop(runtime_key, None)
            config["sample_calls"] = {
                "args": calls.args[-1] if calls.args else None,
                "kwargs": call_kwargs,
            }
        elif include_sample_calls.lower() == "all":
            call_history = calls.to_dict(list_to_dict=True)
            for call_kwargs in call_history["kwargs"].values():
                for runtime_key in (
                    "callbacks",
                    "checkpoint_callback",
                    "resume_from",
                ):
                    call_kwargs.pop(runtime_key, None)
            config["sample_calls"] = call_history
        else:
            raise ValueError(
                "include_sample_calls must be 'last', 'all', or False"
            )
        return config

    def _array_samples(self, particles) -> Samples:
        return Samples(
            particles,
            parameters=self.parameters,
            xp=array_namespace(particles),
        )

    def _log_prior_array(self, particles):
        return self.log_prior(self._array_samples(particles))

    def _log_likelihood_array(self, particles):
        return self.log_likelihood(self._array_samples(particles))

    def _log_posterior_array(self, particles):
        return self._log_prior_array(particles) + self._log_likelihood_array(
            particles
        )

    def _proposal(self):
        return self.prior_proposal

    def _particle_transform(self):
        return self.preconditioning_transform

    def _checkpoint_writers(self, file_path, every, callback=None):
        every = every or 1
        writers = []
        if file_path is not None:
            writers.append(HDF5CheckpointWriter(file_path, every=every))
        if callback is not None:
            writers.append(CallbackCheckpointWriter(callback, every=every))
        return writers

    def _initial_state(self, n_particles, posterior=None, *, rng=None):
        state = draw_initial_state(
            self._proposal(),
            n_particles,
            posterior=posterior,
            rng=rng,
            xp=self.xp,
        )
        return self._validate_initial_state(state)

    def _validate_initial_state(self, state):
        self.prior_proposal._validate(state.particles)
        return state

    def _mcmc_initial_state(self, n_particles, *, rng=None):
        state = draw_mcmc_initial_state(
            self._proposal(),
            n_particles,
            self._log_posterior_array,
            rng=rng,
            xp=self.xp,
        )
        return self._validate_initial_state(state)


__all__ = [
    "CallbackCheckpointWriter",
    "Sampler",
    "load_checkpoint",
]
