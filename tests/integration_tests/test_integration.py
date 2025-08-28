import pytest

from poppy import Poppy


def test_integration_zuko(
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
    if samples_backend == "jax":
        pytest.xfail(
            reason="Converting jax arrays to PyTorch tensors is not supported. See https://github.com/pytorch/pytorch/issues/32868."
        )
    if sampler_config.sampler == "blackjax_smc":
        pytest.xfail(reason="BlackJAX requires JAX arrays.")

    poppy = Poppy(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        parameters=parameters,
        prior_bounds=prior_bounds,
        flow_matching=False,
        bounded_to_unbounded=bounded_to_unbounded,
        flow_backend="zuko",
    )
    poppy.fit(samples, n_epochs=5)
    poppy.sample_posterior(
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

    poppy = Poppy(
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
    poppy.fit(samples, max_epochs=5)
    poppy.sample_posterior(
        n_samples=100,
        sampler=sampler_config.sampler,
        **sampler_config.sampler_kwargs,
    )
