import numpy as np

from aspire import Aspire
from aspire.samples import Samples


def test_resume_from_file_smc(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
):
    dims = 2
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )

    init_samples = Samples(np.random.normal(size=(50, dims)))
    aspire.fit(init_samples, n_epochs=1)

    checkpoint_file = tmp_path / "ckpt.h5"
    with aspire.auto_checkpoint(checkpoint_file, every=1):
        samples = aspire.sample_posterior(
            n_samples=20,
            sampler="smc",
            n_final_samples=20,
            sampler_kwargs={"n_steps": 2},
        )

    resumed = Aspire.resume_from_file(
        checkpoint_file,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    with resumed.auto_checkpoint(checkpoint_file, every=1):
        resumed_samples = resumed.sample_posterior(
            sampler="smc",
        )

    assert len(resumed_samples.x) == len(samples.x)


def test_resume_from_file_manual_call(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
):
    dims = 2
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )

    init_samples = Samples(np.random.normal(size=(50, dims)))
    aspire.fit(init_samples, n_epochs=1)

    checkpoint_file = tmp_path / "ckpt_manual.h5"
    with aspire.auto_checkpoint(checkpoint_file, every=1):
        aspire.sample_posterior(
            n_samples=10,
            sampler="smc",
            n_final_samples=10,
            sampler_kwargs={"n_steps": 1},
        )

    resumed = Aspire.resume_from_file(
        checkpoint_file,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    # Manually call sample_posterior without specifying checkpoint args; defaults should be primed
    resumed_samples = resumed.sample_posterior(sampler="smc")
    assert len(resumed_samples.x) == 10
