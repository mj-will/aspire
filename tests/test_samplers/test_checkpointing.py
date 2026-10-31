import pickle
from pathlib import Path

import h5py
import numpy as np
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


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".PKL"])
@pytest.mark.parametrize("as_string", [False, True])
def test_load_native_sequince_pickle_checkpoint(
    sampler_kwargs, tmp_path: Path, suffix: str, as_string: bool
) -> None:
    checkpoints = []
    sampler = SMCSampler(**sampler_kwargs)
    sampler.sample(
        12,
        max_beta_step=0.2,
        max_n_steps=1,
        sampler_kwargs={"n_steps": 1},
        rng=11,
        checkpoint_callback=checkpoints.append,
    )
    path = tmp_path / f"native{suffix}"
    checkpoints[-1].save_pickle(path)
    checkpoint = load_checkpoint(str(path) if as_string else path)
    assert checkpoint.next_iteration == checkpoints[-1].next_iteration
    np.testing.assert_array_equal(
        checkpoint.state.particles, checkpoints[-1].state.particles
    )
    resume_data = Aspire._load_resume_data(
        path, "checkpoint", "state", "aspire_config"
    )
    assert isinstance(resume_data[0], SMCCheckpoint)
    assert resume_data[2:5] == (None, None, 12)
    with pytest.raises(ValueError, match="sampler state only"):
        Aspire.resume_from_file(
            str(path),
            log_prior=sampler_kwargs["log_prior"],
            log_likelihood=sampler_kwargs["log_likelihood"],
        )


@pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
@pytest.mark.parametrize("automatic", [False, True])
def test_aspire_pickle_checkpoint_resume(
    sampler_kwargs, tmp_path: Path, suffix: str, automatic: bool
) -> None:
    def make_aspire() -> Aspire:
        return Aspire(
            log_prior=sampler_kwargs["log_prior"],
            log_likelihood=sampler_kwargs["log_likelihood"],
            dims=2,
            parameters=sampler_kwargs["parameters"],
            flow=type(sampler_kwargs["proposal"])(),
            xp=sampler_kwargs["xp"],
        )

    path = tmp_path / f"aspire{suffix}"
    options = dict(
        sampler="smc", max_beta_step=0.2, sampler_kwargs={"n_steps": 1}
    )
    full = make_aspire()
    expected = full.sample_posterior(12, rng=11, **options)
    first = make_aspire()
    if automatic:
        with first.auto_checkpoint(path):
            first.sample_posterior(12, rng=11, max_n_steps=1, **options)
    else:
        first.sample_posterior(
            12, rng=11, max_n_steps=1, checkpoint_path=path, **options
        )

    assert SMCCheckpoint.load_pickle(path).next_iteration == 1
    resumed = make_aspire()
    if automatic:
        with resumed.auto_checkpoint(path, resume=True):
            assert isinstance(resumed._resume_from_default, SMCCheckpoint)
            samples = resumed.sample_posterior(**options)
    else:
        samples = resumed.sample_posterior(
            12, resume_from=str(path), checkpoint_path=path, **options
        )
    assert samples.beta == 1.0
    np.testing.assert_array_equal(samples.x, expected.x)
    assert samples.log_evidence == expected.log_evidence
    assert SMCCheckpoint.load_pickle(path).next_iteration > 1
