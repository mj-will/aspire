from aspire import Aspire


def test_resume_from_file_smc(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    samples,
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

    aspire.fit(samples, n_epochs=10)

    checkpoint_file = tmp_path / "ckpt.h5"
    with aspire.auto_checkpoint(checkpoint_file, every=1):
        samples = aspire.sample_posterior(
            n_samples=20,
            sampler="smc",
            n_final_samples=25,
            sampler_kwargs={"n_steps": 10, "step_fn": "pcn"},
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

    assert len(resumed_samples.x) == 25


def test_resume_from_file_manual_call(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    samples,
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

    aspire.fit(samples, n_epochs=10)

    checkpoint_file = tmp_path / "ckpt_manual.h5"
    with aspire.auto_checkpoint(checkpoint_file, every=1):
        aspire.sample_posterior(
            n_samples=20,
            sampler="smc",
            n_final_samples=25,
            sampler_kwargs={"n_steps": 10, "step_fn": "pcn"},
        )

    resumed = Aspire.resume_from_file(
        checkpoint_file,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    # Manually call sample_posterior without specifying checkpoint args; defaults should be primed
    resumed_samples = resumed.sample_posterior(sampler="smc")
    assert len(resumed_samples.x) == 25


def test_auto_checkpoint_resume_same_instance(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    samples,
):
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )
    aspire.fit(samples, n_epochs=10)

    checkpoint_file = tmp_path / "ckpt_same_instance.h5"
    with aspire.auto_checkpoint(checkpoint_file, every=1):
        aspire.sample_posterior(
            n_samples=20,
            sampler="smc",
            n_final_samples=25,
            sampler_kwargs={"n_steps": 10, "step_fn": "pcn"},
        )

    assert not hasattr(aspire, "_resume_from_default")
    with aspire.auto_checkpoint(checkpoint_file, every=1, resume=True):
        assert aspire._resume_n_samples == 20
        resumed_samples = aspire.sample_posterior(sampler="smc")

    assert len(resumed_samples.x) == 25
    assert not hasattr(aspire, "_resume_from_default")


def test_auto_checkpoint_resume_loads_flow_for_new_instance(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    samples,
):
    checkpoint_file = tmp_path / "ckpt_new_instance.h5"

    writer = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )
    writer.fit(samples, checkpoint_path=checkpoint_file, n_epochs=10)
    with writer.auto_checkpoint(checkpoint_file, every=1):
        writer.sample_posterior(
            n_samples=20,
            sampler="smc",
            n_final_samples=25,
            sampler_kwargs={"n_steps": 10, "step_fn": "pcn"},
        )

    resumed = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )
    assert resumed.flow is None

    with resumed.auto_checkpoint(checkpoint_file, every=1, resume=True):
        assert resumed.flow is not None
        resumed_samples = resumed.sample_posterior(sampler="smc")

    assert len(resumed_samples.x) == 25


def test_auto_checkpoint_resume_skips_flow_training(
    tmp_path,
    log_likelihood,
    log_prior,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    samples,
):
    checkpoint_file = tmp_path / "ckpt_skip_fit.h5"

    writer = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )
    writer.fit(samples, checkpoint_path=checkpoint_file, n_epochs=10)

    resumed = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=2,
        parameters=parameters,
        prior_bounds=prior_bounds,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )

    with resumed.auto_checkpoint(checkpoint_file, resume=True):
        original_fit = resumed.flow.fit

        def fail_fit(*args, **kwargs):
            raise AssertionError("flow.fit should not be called")

        resumed.flow.fit = fail_fit
        history = resumed.fit(samples, n_epochs=10)
        resumed.flow.fit = original_fit

        assert history.training_loss == []
        assert history.validation_loss == []
