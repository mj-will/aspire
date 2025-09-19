import pytest
from aspire import Aspire


def test_integration_zuko(
    log_likelihood,
    log_prior,
    dims,
    samples,
    parameters,
    prior_bounds,
    bounded_to_unbounded,
    sampler_config,
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
    )
    aspire.fit(samples, n_epochs=5)
    aspire.sample_posterior(
        n_samples=100,
        sampler=sampler_config.sampler,
        **sampler_config.sampler_kwargs,
    )


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
):
    import jax

    if sampler_config.sampler == "blackjax_smc" and samples_backend != "jax":
        pytest.xfail(reason="BlackJAX requires JAX arrays.")

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
    )
    aspire.fit(samples, max_epochs=5)
    aspire.sample_posterior(
        n_samples=100,
        sampler=sampler_config.sampler,
        **sampler_config.sampler_kwargs,
    )
