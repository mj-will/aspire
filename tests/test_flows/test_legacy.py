"""Exercise legacy signatures, fitted state, and proposal densities."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from sequince.flows import clone_flow

from aspire import Aspire, LegacyFlowAdapter
from aspire.compat import _adapt_flow
from aspire.history import FlowHistory
from aspire.samplers.base import _ValidatedFlowProposal


class OldFlow:
    """A fitted affine Gaussian using the former flow interface."""

    xp = np

    def __init__(self, dims: int = 2) -> None:
        self.dims = dims
        self._init_args = {"dims": dims}
        self.mean = np.zeros(dims)
        self.rng = np.random.default_rng(42)

    def fit(self, x: np.ndarray, **kwargs: Any) -> FlowHistory:
        assert kwargs == {"epochs": 2}
        self.mean = x.mean(axis=0)
        return FlowHistory(training_loss=[2.0, 1.0])

    def sample(self, n: int) -> np.ndarray:
        return self.rng.normal(size=(n, self.dims)) * 2 + self.mean

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        z, jac = self.forward(x)
        return -0.5 * np.sum(z**2 + np.log(2 * np.pi), axis=-1) + jac

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (x - self.mean) / 2, np.full(len(x), -self.dims * np.log(2))

    def inverse(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z * 2 + self.mean, np.full(len(z), self.dims * np.log(2))

    def save(self, handle: h5py.File, path: str = "flow") -> None:
        handle.create_dataset(path + "/mean", data=self.mean)

    @classmethod
    def load(cls, handle: h5py.File, path: str = "flow") -> "OldFlow":
        flow = cls(len(handle[path + "/mean"]))
        flow.mean = handle[path + "/mean"][()]
        return flow


def test_training_density_and_transforms() -> None:
    flow = LegacyFlowAdapter(OldFlow())
    assert flow.fit(np.full((10, 2), 3.0), epochs=2, rng=123) is flow
    assert flow.fit_history["training_loss"] == [2.0, 1.0]
    x, log_q = flow.sample_and_log_prob(8, rng=123)
    np.testing.assert_allclose(log_q, flow.log_prob(x))
    z, jac = flow.forward(x)
    restored, inverse_jac = flow.inverse(z)
    np.testing.assert_allclose(restored, x)
    np.testing.assert_allclose(jac + inverse_jac, 0)
    # Cloning builds an unfitted model for Sequince's training path.
    assert isinstance(clone_flow(flow), LegacyFlowAdapter)


def test_auto_adaptation_for_aspire_and_sampler() -> None:
    old = OldFlow()
    with pytest.warns(DeprecationWarning, match="adapted automatically"):
        analysis = Aspire(
            log_likelihood=None, log_prior=None, dims=2, flow=old
        )
    assert isinstance(analysis.flow, LegacyFlowAdapter)
    with pytest.warns(DeprecationWarning, match="adapted automatically"):
        proposal = _ValidatedFlowProposal(old, dims=2)
    x, log_q = proposal.sample_and_log_prob(5, rng=23, xp=np)
    np.testing.assert_allclose(log_q, old.log_prob(x))
    with pytest.warns(DeprecationWarning, match="adapted automatically"):
        analysis.flow = old
    assert analysis.flow.legacy_flow is old
    assert _adapt_flow(analysis.flow) is analysis.flow


def test_saved_flow_preserves_fitted_density(tmp_path: Path) -> None:
    flow = LegacyFlowAdapter(OldFlow())
    flow.fit(np.full((10, 2), 4.0), epochs=2)
    writer = Aspire(log_likelihood=None, log_prior=None, dims=2, flow=flow)
    x = flow.sample(10)
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        writer.save_flow(handle)
    restored = Aspire(
        log_likelihood=None,
        log_prior=None,
        dims=2,
        flow=LegacyFlowAdapter(OldFlow()),
    )
    with h5py.File(path, "r") as handle:
        restored.load_flow(handle)
    np.testing.assert_allclose(restored.flow.log_prob(x), flow.log_prob(x))
    assert restored.flow.fit_history == flow.fit_history


def test_joint_sampling_and_explicit_rng() -> None:
    class JointFlow(OldFlow):
        def sample_and_log_prob(
            self, n: int, *, rng: Any = None
        ) -> tuple[np.ndarray, np.ndarray]:
            x = rng.normal(size=(n, self.dims))
            return x, self.log_prob(x)

    wrapped = LegacyFlowAdapter(JointFlow())
    x, log_q = wrapped.sample_and_log_prob(3, rng=np.random.default_rng(1))
    expected = np.random.default_rng(1).normal(size=(3, 2))
    np.testing.assert_allclose(x, expected)
    np.testing.assert_allclose(log_q, wrapped.log_prob(x))


def test_errors_are_not_swallowed() -> None:
    class BrokenFlow(OldFlow):
        def sample(self, n: int) -> np.ndarray:
            raise TypeError("failure inside flow")

    with pytest.raises(TypeError, match="failure inside flow"):
        LegacyFlowAdapter(BrokenFlow()).sample(1)
    flow = LegacyFlowAdapter(OldFlow())
    del flow._init_args
    with pytest.raises(ValueError, match="flow_kwargs"):
        flow.state_dict()


def test_native_state_methods_and_modern_identity() -> None:
    class StateFlow(OldFlow):
        def state_dict(self) -> dict[str, Any]:
            return {"mean": self.mean.copy()}

        def load_state_dict(self, state: dict[str, Any]) -> None:
            self.mean = state["mean"]

    fitted = LegacyFlowAdapter(StateFlow())
    fitted.fit(np.ones((3, 2)), epochs=2)
    restored = LegacyFlowAdapter(StateFlow())
    restored.load_state_dict(fitted.state_dict())
    np.testing.assert_allclose(
        restored.log_prob(np.zeros((2, 2))), fitted.log_prob(np.zeros((2, 2)))
    )

    class ModernFlow:
        def sample(
            self, n: int, context: Any = None, *, rng: Any = None
        ) -> None:
            pass

        def log_prob(self, x: np.ndarray, context: Any = None) -> None:
            pass

    modern = ModernFlow()
    assert _adapt_flow(modern) is modern


def test_importance_weights_use_legacy_density() -> None:
    from aspire.samplers.importance import ImportanceSampler

    old = OldFlow()
    sampler = ImportanceSampler(
        proposal=LegacyFlowAdapter(old),
        log_likelihood=lambda samples: old.log_prob(samples.x),
        log_prior=lambda samples: np.zeros(len(samples)),
        dims=2,
        xp=np,
    )
    samples = sampler.sample(32, rng=123)
    np.testing.assert_allclose(samples.log_q, old.log_prob(samples.x))
    np.testing.assert_allclose(samples.log_likelihood - samples.log_q, 0)
