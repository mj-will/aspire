"""Verify Aspire's persistence envelope for both density flow backends."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from aspire import Aspire, Samples
from aspire.utils import to_numpy


@pytest.mark.parametrize("backend", ["zuko", "flowjax"])
def test_flow_saved_state_preserves_density(
    tmp_path: Path, backend: str
) -> None:
    pytest.importorskip(backend)

    def log_density(samples: Samples) -> np.ndarray:
        return -0.5 * np.sum(to_numpy(samples.x) ** 2, axis=-1)

    options = dict(
        log_prior=log_density,
        log_likelihood=log_density,
        dims=2,
        parameters=["x", "y"],
        prior_bounds={"x": (-5, 5), "y": (-5, 5)},
        flow_backend=backend,
    )
    writer = Aspire(**options)
    training = Samples(np.random.default_rng(12).normal(size=(64, 2)))
    writer.fit(
        training,
        **({"n_epochs": 2} if backend == "zuko" else {"max_epochs": 2}),
    )
    particles = writer.sample_flow(16).x
    expected = to_numpy(writer.flow.log_prob(particles))
    path = tmp_path / "flow.h5"
    with h5py.File(path, "w") as handle:
        writer.save_flow(handle)
    restored = Aspire(**options)
    with h5py.File(path, "r") as handle:
        restored.load_flow(handle)
    np.testing.assert_allclose(
        to_numpy(restored.flow.log_prob(particles)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
