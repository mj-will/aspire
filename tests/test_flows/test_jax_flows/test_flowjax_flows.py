import jax
import jax.numpy as jnp
from aspire.flows.jax.flows import FlowJax
from aspire.transforms import FlowTransform


def test_zuko_flow():
    dims = 3
    parameters = [f"x_{i}" for i in range(dims)]

    data_transform = FlowTransform(parameters=parameters, xp=jnp)
    key = jax.random.key(42)
    key, flow_key = jax.random.split(key)

    # Create an instance of FlowJax
    flow = FlowJax(
        dims=dims,
        key=flow_key,
        device="cpu",
        data_transform=data_transform,
    )

    key, samples_key = jax.random.split(key)
    x = jax.random.normal(samples_key, (100, dims))

    flow.fit_data_transform(x)

    # Check if the flow is initialized correctly
    assert flow.dims == dims

    # Check if the flow is an instance of ZukoFlow
    assert isinstance(flow, FlowJax)

    x = jnp.array([0.1, 0.2, 0.3])

    log_prob = flow.log_prob(x)

    assert log_prob.shape == (1,)

    key, sample_key = jax.random.split(key)
    x = flow.sample(1)
    assert x.shape == (1, dims)
