from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from aspire import Aspire, Samples
from aspire.flows.base import Flow
from aspire.history import FitHistory
from aspire.proposals import GaussianProposal, Proposal
from aspire.utils import load_from_h5_file


def log_likelihood(samples):
    return -0.5 * np.sum(samples.x**2, axis=1)


def log_prior(samples):
    return np.zeros(len(samples))


@pytest.mark.parametrize(
    ("dims", "n_samples"),
    [(1, 1), (1, 5), (2, 1), (2, 5)],
)
def test_gaussian_proposal_sample_shapes(dims, n_samples):
    proposal = GaussianProposal(
        dims=dims,
        mean=np.zeros(dims),
        cov=np.eye(dims),
    )

    samples, log_prob = proposal.sample_and_log_prob(n_samples)

    assert samples.shape == (n_samples, dims)
    assert log_prob.shape == (n_samples,)
    np.testing.assert_allclose(log_prob, proposal.log_prob(samples))


def test_gaussian_proposal_fit_and_validation():
    proposal = GaussianProposal(dims=1)
    proposal.fit(np.array([[1.0], [2.0], [3.0]]))

    np.testing.assert_allclose(proposal.mean, [2.0])
    np.testing.assert_allclose(proposal.cov, [[1.0]])

    with pytest.raises(ValueError, match="At least two samples"):
        proposal.fit(np.array([[1.0]]))
    with pytest.raises(ValueError, match="mean and cov"):
        GaussianProposal(dims=1, mean=[0.0])


def test_gaussian_proposal_log_prob():
    proposal = GaussianProposal(dims=2, mean=np.zeros(2), cov=np.eye(2))

    log_prob = proposal.log_prob(np.array([[0.0, 0.0], [1.0, 1.0]]))

    np.testing.assert_allclose(
        log_prob,
        [-np.log(2.0 * np.pi), -np.log(2.0 * np.pi) - 1.0],
    )


def test_proposal_protocol_requires_only_sampling_interface():
    class MinimalProposal:
        xp = np

        def sample_and_log_prob(self, n_samples):
            return np.zeros((n_samples, 1)), np.zeros(n_samples)

        def log_prob(self, x):
            return np.zeros(len(x))

    assert isinstance(MinimalProposal(), Proposal)


def test_aspire_fit_leaves_non_trainable_proposal_unchanged():
    class FixedProposal:
        xp = np

        def sample_and_log_prob(self, n_samples):
            return np.zeros((n_samples, 1)), np.zeros(n_samples)

        def log_prob(self, x):
            return np.zeros(len(x))

    proposal = FixedProposal()
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=1,
        proposal=proposal,
    )

    history = aspire.fit(Samples(np.array([[0.0], [1.0]])))

    assert aspire.proposal is proposal
    assert isinstance(history, FitHistory)


def test_aspire_fit_returns_history_when_proposal_fit_returns_none():
    proposal = GaussianProposal(dims=1)
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=1,
        proposal=proposal,
    )

    history = aspire.fit(Samples(np.array([[1.0], [2.0], [3.0]])))

    assert isinstance(history, FitHistory)


def test_sample_posterior_requires_initialized_proposal():
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=1,
    )

    with pytest.raises(RuntimeError, match="before initializing the proposal"):
        aspire.sample_posterior(1)


def test_save_proposal_requires_save_method(tmp_path):
    class MinimalProposal:
        xp = np

        def sample_and_log_prob(self, n_samples):
            return np.zeros((n_samples, 1)), np.zeros(n_samples)

        def log_prob(self, x):
            return np.zeros(len(x))

    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=1,
        proposal=MinimalProposal(),
    )

    with h5py.File(tmp_path / "proposal.h5", "w") as h5_file:
        with pytest.raises(
            ValueError, match="does not implement a 'save' method"
        ):
            aspire.save_proposal(h5_file)


def test_gaussian_proposal_save_load_preserves_frozen(tmp_path):
    proposal = GaussianProposal(
        dims=2,
        mean=np.zeros(2),
        cov=np.eye(2),
        frozen=True,
    )
    file_path = tmp_path / "proposal.h5"

    with h5py.File(file_path, "w") as h5_file:
        proposal.save(h5_file)
    with h5py.File(file_path, "r") as h5_file:
        loaded = GaussianProposal.load(h5_file)

    assert loaded.frozen is True
    np.testing.assert_array_equal(loaded.mean, proposal.mean)
    np.testing.assert_array_equal(loaded.cov, proposal.cov)


def test_aspire_load_proposal_infers_saved_class(tmp_path):
    proposal = GaussianProposal(2, mean=np.zeros(2), cov=np.eye(2))
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        proposal=proposal,
    )
    file_path = tmp_path / "proposal.h5"

    with h5py.File(file_path, "w") as h5_file:
        aspire.save_proposal(h5_file)

    loaded = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
    )
    with h5py.File(file_path, "r") as h5_file:
        loaded.load_proposal(h5_file)

    assert isinstance(loaded.proposal, GaussianProposal)
    np.testing.assert_array_equal(loaded.proposal.mean, proposal.mean)


def test_aspire_load_proposal_accepts_explicit_class(tmp_path):
    proposal = GaussianProposal(2, mean=np.zeros(2), cov=np.eye(2))
    file_path = tmp_path / "proposal.h5"
    with h5py.File(file_path, "w") as h5_file:
        proposal.save(h5_file)

    loaded = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
    )
    with h5py.File(file_path, "r") as h5_file:
        loaded.load_proposal(
            h5_file,
            proposal_class=GaussianProposal,
        )

    assert isinstance(loaded.proposal, GaussianProposal)


def test_resume_from_file_restores_gaussian_proposal(tmp_path):
    proposal = GaussianProposal(2, mean=np.zeros(2), cov=np.eye(2))
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        proposal=proposal,
    )
    file_path = tmp_path / "checkpoint.h5"
    with h5py.File(file_path, "w") as h5_file:
        aspire.save_config(h5_file, include_sampler_config=False)
        aspire.save_proposal(h5_file)
        # Simulate a file saved before proposal class metadata was added.
        del h5_file["proposal"].attrs["proposal_module"]
        del h5_file["proposal"].attrs["proposal_qualname"]

    resumed = Aspire.resume_from_file(
        file_path,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    assert isinstance(resumed.proposal, GaussianProposal)
    np.testing.assert_array_equal(resumed.proposal.mean, proposal.mean)


def test_aspire_config_uses_safe_proposal_metadata(tmp_path):
    class DummyFlow(Flow):
        xp = np

        def save(self, h5_file, path="proposal"):
            h5_file.require_group(path)

    flow = DummyFlow(dims=2, device=None)
    flow.config_dict = Mock(
        side_effect=AssertionError("runtime flow config must not be embedded")
    )
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        proposal=flow,
    )
    file_path = tmp_path / "config.h5"

    with h5py.File(file_path, "w") as h5_file:
        aspire.save_config(h5_file, include_sampler_config=False)
        config = load_from_h5_file(h5_file, "aspire_config")

    assert config["proposal_config"] == {
        "proposal_class": "DummyFlow",
        "proposal_module": DummyFlow.__module__,
    }
    flow.config_dict.assert_not_called()


def test_auto_checkpoint_can_skip_proposal(tmp_path):
    proposal = GaussianProposal(2, mean=np.zeros(2), cov=np.eye(2))
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        proposal=proposal,
    )
    file_path = tmp_path / "checkpoint.h5"

    with aspire.auto_checkpoint(
        file_path,
        save_config=False,
        save_proposal=False,
    ):
        aspire.fit(
            Samples(
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [2.0, 1.0],
                        [1.0, 2.0],
                    ]
                )
            )
        )
        aspire.sample_posterior(4, sampler="importance")

    with h5py.File(file_path, "r") as h5_file:
        assert "proposal" not in h5_file


def test_deprecated_flow_compatibility(monkeypatch, tmp_path):
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
    )
    init_flow = Mock()
    monkeypatch.setattr(aspire, "_init_flow", init_flow)

    with pytest.warns(DeprecationWarning, match="init_flow"):
        aspire.init_flow()
    init_flow.assert_called_once_with()

    with (
        pytest.warns(DeprecationWarning, match="save_flow"),
        aspire.auto_checkpoint(
            tmp_path / "checkpoint.h5",
            save_flow=False,
        ),
    ):
        assert aspire._checkpoint_defaults["save_proposal"] is False


def test_init_proposal_is_idempotent(monkeypatch):
    proposal = GaussianProposal(2, mean=np.zeros(2), cov=np.eye(2))
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        proposal=proposal,
    )
    init_flow = Mock()
    monkeypatch.setattr(aspire, "_init_flow", init_flow)

    assert aspire.init_proposal() is proposal
    init_flow.assert_not_called()
