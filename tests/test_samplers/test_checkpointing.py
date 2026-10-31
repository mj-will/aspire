import pickle

import h5py
import pytest
from sequince.checkpoint import SMCCheckpoint
from sequince.transforms import IdentityTransform

from aspire.aspire import Aspire
from aspire.samplers.base import load_checkpoint
from aspire.samplers.smc import SMCSampler
from aspire.utils import recursively_save_to_h5_file


@pytest.mark.parametrize("as_bytes", [False, True])
def test_legacy_checkpoint_payload_is_rejected(as_bytes: bool) -> None:
    payload = {"samples": None, "sampler": "smc"}
    with pytest.raises(ValueError, match="Legacy Aspire checkpoints"):
        load_checkpoint(pickle.dumps(payload) if as_bytes else payload)


def test_legacy_saved_flow_is_rejected(sampler_kwargs, tmp_path) -> None:
    aspire = Aspire(
        log_prior=sampler_kwargs["log_prior"],
        log_likelihood=sampler_kwargs["log_likelihood"],
        dims=2,
    )
    with h5py.File(tmp_path / "legacy.h5", "w") as handle:
        handle.create_group("flow")
        with pytest.raises(ValueError, match="Legacy Aspire saved flows"):
            aspire.load_flow(handle)


def test_legacy_checkpoint_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("checkpoint/state", data=[0], dtype="u1")
    with pytest.raises(ValueError, match="Legacy Aspire checkpoints"):
        Aspire._load_resume_data(path, "checkpoint", "state", "aspire_config")
    with pytest.raises(ValueError, match="Legacy Aspire checkpoints"):
        load_checkpoint(path)


def test_smc_checkpoint_uses_sequince_format(sampler_kwargs, tmp_path):
    checkpoint_path = tmp_path / "smc.h5"
    sampler_kwargs["preconditioning_transform"] = IdentityTransform()
    sampler = SMCSampler(**sampler_kwargs)
    sampler.sample(
        12,
        adaptive=False,
        n_steps=2,
        sampler_kwargs={"n_steps": 1},
        rng=9,
        checkpoint_file_path=checkpoint_path,
        checkpoint_every=1,
    )

    with h5py.File(checkpoint_path, "r") as h5_file:
        checkpoint = SMCCheckpoint.load_hdf5(h5_file)
    with h5py.File(checkpoint_path, "a") as h5_file:
        recursively_save_to_h5_file(
            h5_file,
            "sampler_config",
            {"sampler_type": "smc"},
        )
    resume_data = Aspire._load_resume_data(
        checkpoint_path,
        checkpoint_path="checkpoint",
        checkpoint_dset="state",
        config_path="aspire_config",
    )

    assert checkpoint.next_iteration == 2
    assert checkpoint.state.particles.shape == (12, 2)
    assert checkpoint.initial_state is not None
    assert checkpoint.initial_state.particles.shape == (12, 2)
    assert len(checkpoint.states) == 2
    assert resume_data[0] is not None
    assert resume_data[3] == "smc"
    assert resume_data[4] == 12


def test_smc_resumes_native_checkpoint(sampler_kwargs, tmp_path, caplog):
    checkpoint_path = tmp_path / "resume.h5"
    first_sampler = SMCSampler(**sampler_kwargs)
    partial = first_sampler.sample(
        12,
        max_beta_step=0.2,
        max_n_steps=1,
        sampler_kwargs={"n_steps": 1},
        rng=11,
        checkpoint_file_path=checkpoint_path,
    )
    assert partial.beta == pytest.approx(0.2)
    assert "before reaching the posterior" in caplog.text

    resumed_sampler = SMCSampler(**sampler_kwargs)
    resumed = resumed_sampler.sample(
        12,
        max_beta_step=0.2,
        max_n_steps=10,
        sampler_kwargs={"n_steps": 1},
        resume_from=checkpoint_path,
    )

    assert resumed.beta == 1.0
    assert len(resumed_sampler.result.states) > 1
    assert resumed_sampler.result.initial_state is not None
    assert len(resumed_sampler.history.sample_history) == (
        len(resumed_sampler.result.states) + 1
    )
