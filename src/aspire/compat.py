"""Compatibility adapters for externally supplied Aspire flows."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from io import BytesIO
from typing import TYPE_CHECKING, Any, Protocol

import h5py

if TYPE_CHECKING:
    from sequince.flows import NormalizingFlow
    from sequince.steps.context import StepContext
    from sequince.typing import Array
    from sequince.utils.random import RNGLike

    from .history import FlowHistory


class LegacyFlow(Protocol):
    """Sampling and training interface used by external pre-migration flows."""

    xp: Any

    def sample(self, n_samples: int) -> Array: ...

    def log_prob(self, particles: Array) -> Array: ...

    def fit(self, particles: Array, **kwargs: Any) -> FlowHistory | None: ...


class LegacyFlowAdapter:
    """Expose an old-style Aspire flow through Sequince's flow interface.

    Parameters
    ----------
    flow : LegacyFlow or None, optional
        Existing external flow. Its density methods must already include any
        data transforms and their Jacobians.
    flow_class : type or None, optional
        Legacy class used to reconstruct a saved adapter. Normally inferred
        from ``flow``.
    flow_kwargs : mapping or None, optional
        Constructor arguments for reconstruction. If omitted, use the flow's
        ``_init_args`` or ``config_dict()`` when available. Persistence requires
        reconstruction arguments and either state-dict methods or ``save``
        and ``load`` methods.

    Notes
    -----
    Only explicitly declared ``context`` and ``rng`` parameters receive those
    values; they are not injected into a legacy trainer's arbitrary kwargs.
    Methods without an RNG argument retain their original RNG behavior.
    Particle arrays have shape ``(n_samples, dims)``; log densities and
    log Jacobians have shape ``(n_samples,)``. Arrays retain the wrapped
    flow's backend, dtype, and device.
    Flows without forward/inverse methods can be used as sampling proposals,
    but cannot provide a flow preconditioner.
    """

    def __init__(
        self,
        flow: LegacyFlow | None = None,
        *,
        flow_class: type[LegacyFlow] | None = None,
        flow_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if flow is None:
            if flow_class is None or flow_kwargs is None:
                raise TypeError(
                    "Supply a flow or its class and constructor arguments."
                )
            arguments = dict(flow_kwargs)
            extra = arguments.pop("kwargs", {})
            flow = flow_class(**extra, **arguments)
        if getattr(flow, "xp", None) is None:
            raise ValueError("Legacy flow must define an xp namespace.")
        self.legacy_flow = flow
        self.xp = flow.xp
        self.device = getattr(flow, "device", None)
        self.dtype = getattr(flow, "dtype", None)
        self.dims = getattr(flow, "dims", None)
        self.fit_history: dict[str, list[float]] = {}
        if flow_kwargs is None:
            flow_kwargs = getattr(flow, "_init_args", None)
            if flow_kwargs is None:
                config = getattr(flow, "config_dict", None)
                if callable(config):
                    flow_kwargs = config()
        if isinstance(flow_kwargs, Mapping):
            self._init_args = {
                "flow_class": type(flow),
                "flow_kwargs": dict(flow_kwargs),
            }

    def _call(
        self,
        method: str,
        value: Any,
        *,
        context: StepContext | None = None,
        rng: RNGLike = None,
        **kwargs: Any,
    ) -> Any:
        function = getattr(self.legacy_flow, method, None)
        if not callable(function):
            raise NotImplementedError(
                f"Legacy flow does not implement {method}."
            )
        parameters = inspect.signature(function).parameters
        for name, option in (("context", context), ("rng", rng)):
            parameter = parameters.get(name)
            if parameter is not None and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                kwargs[name] = option
        return function(value, **kwargs)

    def fit(
        self,
        particles: Array,
        context: StepContext | None = None,
        *,
        rng: RNGLike = None,
        **kwargs: Any,
    ) -> LegacyFlowAdapter:
        """Train the legacy flow and retain its loss history."""
        history = self._call(
            "fit", particles, context=context, rng=rng, **kwargs
        )
        self.fit_history = {
            "training_loss": list(getattr(history, "training_loss", [])),
            "validation_loss": list(getattr(history, "validation_loss", [])),
        }
        return self

    def sample(
        self,
        n_samples: int = 1,
        context: StepContext | None = None,
        *,
        rng: RNGLike = None,
    ) -> Array:
        """Draw from the legacy flow using the keywords it supports."""
        return self._call("sample", n_samples, context=context, rng=rng)

    def log_prob(
        self, particles: Array, context: StepContext | None = None
    ) -> Array:
        """Evaluate the complete legacy density in physical coordinates."""
        return self._call("log_prob", particles, context=context)

    def sample_and_log_prob(
        self,
        n_samples: int = 1,
        context: StepContext | None = None,
        *,
        rng: RNGLike = None,
    ) -> tuple[Array, Array]:
        """Use a joint legacy draw when available; otherwise evaluate its samples."""
        if callable(getattr(self.legacy_flow, "sample_and_log_prob", None)):
            return self._call(
                "sample_and_log_prob", n_samples, context=context, rng=rng
            )
        particles = self.sample(n_samples, context=context, rng=rng)
        return particles, self.log_prob(particles, context=context)

    def forward(
        self, particles: Array, context: StepContext | None = None
    ) -> tuple[Array, Array]:
        """Delegate the full physical-to-latent map and its Jacobian."""
        return self._call("forward", particles, context=context)

    def inverse(
        self, values: Array, context: StepContext | None = None
    ) -> tuple[Array, Array]:
        """Delegate the full latent-to-physical map and its Jacobian."""
        return self._call("inverse", values, context=context)

    def state_dict(self) -> dict[str, Any]:
        """Persist fitted state through native state methods or legacy HDF5 IO."""
        if not hasattr(self, "_init_args"):
            raise ValueError(
                "Supply flow_kwargs to make this legacy flow reconstructible."
            )
        state = getattr(self.legacy_flow, "state_dict", None)
        restore = getattr(self.legacy_flow, "load_state_dict", None)
        if callable(state) and callable(restore):
            return {"state_dict": state(), "fit_history": self.fit_history}
        save = getattr(self.legacy_flow, "save", None)
        load = getattr(type(self.legacy_flow), "load", None)
        if not callable(save) or not callable(load):
            raise ValueError(
                "Legacy flow must implement state methods or save/load for persistence."
            )
        buffer = BytesIO()
        with h5py.File(buffer, "w") as handle:
            save(handle, path="flow")
        return {"hdf5": buffer.getvalue(), "fit_history": self.fit_history}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the wrapped flow without discarding fitted legacy state."""
        if "state_dict" in state:
            getattr(self.legacy_flow, "load_state_dict")(state["state_dict"])
        elif "hdf5" in state:
            with h5py.File(BytesIO(state["hdf5"]), "r") as handle:
                self.legacy_flow = getattr(type(self.legacy_flow), "load")(
                    handle, path="flow"
                )
        else:
            raise ValueError(
                "Legacy flow checkpoint contains no fitted state."
            )
        self.fit_history = dict(state.get("fit_history", {}))
        self.xp = self.legacy_flow.xp
        self.device = getattr(self.legacy_flow, "device", None)
        self.dtype = getattr(self.legacy_flow, "dtype", None)
        self.dims = getattr(self.legacy_flow, "dims", None)


def _adapt_flow(flow: Any) -> NormalizingFlow:
    """Adapt supplied legacy instances while leaving modern instances unchanged."""
    if isinstance(flow, LegacyFlowAdapter):
        return flow
    for name, keywords in (
        ("sample", ("context", "rng")),
        ("log_prob", ("context",)),
        ("sample_and_log_prob", ("context", "rng")),
        ("forward", ("context",)),
        ("inverse", ("context",)),
    ):
        function = getattr(flow, name, None)
        if not callable(function):
            continue
        parameters = inspect.signature(function).parameters
        accepts_kwargs = any(
            p.kind == p.VAR_KEYWORD for p in parameters.values()
        )
        if not accepts_kwargs and any(
            key not in parameters for key in keywords
        ):
            warnings.warn(
                "An old-style flow was adapted automatically; use LegacyFlowAdapter "
                "explicitly or implement the Sequince flow interface.",
                DeprecationWarning,
                stacklevel=3,
            )
            return LegacyFlowAdapter(flow)
    return flow
