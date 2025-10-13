import pytest

from aspire import Aspire
from aspire.utils import AspireFile


@pytest.fixture(params=[None, "float32", "float64"])
def dtype(request):
    return request.param


def test_integration_zuko(
    log_likelihood,
    log_prior,
    dims,
    samples,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    sampler_config,
    dtype,
    tmp_path,
):
    if sampler_config.sampler == "blackjax_smc":
        pytest.xfail(reason="BlackJAX requires JAX arrays.")

    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        parameters=parameters,
        prior_bounds=prior_bounds,
        flow_matching=False,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
        dtype=dtype,
    )
    aspire.fit(samples, n_epochs=5)
    aspire.sample_posterior(
        n_samples=100,
        sampler=sampler_config.sampler,
        **sampler_config.sampler_kwargs,
    )
    with AspireFile(tmp_path / "test_integration_zuko.h5", "w") as h5_file:
        aspire.save_config(h5_file)
        aspire.save_flow(h5_file)
        samples.save(h5_file, path="posterior_samples")


@pytest.mark.requires("flowjax")
def test_integration_flowjax(
    log_likelihood,
    log_prior,
    dims,
    samples,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    samples_backend,
    sampler_config,
    dtype,
    tmp_path,
):
    import jax

    if sampler_config.sampler == "blackjax_smc":
        if samples_backend != "jax":
            pytest.xfail(reason="BlackJAX requires JAX arrays.")
        if dtype == "float32":
            pytest.xfail(
                reason="BlackJAX tests with float32 fail when running with jax defaults set to float64 dtypes."
            )

    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        parameters=parameters,
        prior_bounds=prior_bounds,
        flow_matching=False,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="flowjax",
        key=jax.random.key(0),
        dtype=dtype,
    )
    aspire.fit(samples, max_epochs=5)
    posterior_samples = aspire.sample_posterior(
        n_samples=100,
        sampler=sampler_config.sampler,
        **sampler_config.sampler_kwargs,
    )

    with AspireFile(tmp_path / "test_integration_flowjax.h5", "w") as h5_file:
        aspire.save_config(h5_file)
        aspire.save_flow(h5_file)
        posterior_samples.save(h5_file, path="posterior_samples")


def test_init_existing_flow(
    log_likelihood,
    log_prior,
    dims,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
):
    aspire_kwargs = {
        "log_likelihood": log_likelihood,
        "log_prior": log_prior,
        "dims": dims,
        "parameters": parameters,
        "prior_bounds": prior_bounds,
        "flow_matching": False,
        "bounded_to_unbounded": bounded_to_unbounded,
        "flow_backend": "zuko",
    }

    aspire = Aspire(**aspire_kwargs)
    aspire.init_flow()

    saved_flow = aspire.flow
    new_aspire_obj = Aspire(**aspire_kwargs | {"flow": saved_flow})

    assert new_aspire_obj.flow is aspire.flow

    new_aspire_obj.flow = saved_flow

    assert new_aspire_obj.flow is saved_flow
