import jax
import jax.numpy as jnp
import pytest

from aspire.flows.jax.flows import FlowJax
from aspire.transforms import FlowTransform
from aspire.utils import AspireFile


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_flowjax_flow(dtype):
    dims = 3
    parameters = [f"x_{i}" for i in range(dims)]

    data_transform = FlowTransform(parameters=parameters, xp=jnp, dtype=dtype)
    key = jax.random.key(42)
    key, flow_key = jax.random.split(key)

    # Create an instance of FlowJax
    flow = FlowJax(
        dims=dims,
        key=flow_key,
        device="cpu",
        data_transform=data_transform,
        dtype=dtype,
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


def test_flowjax_save_and_load(tmp_path):
    key = jax.random.key(42)
    key, flow_key = jax.random.split(key)
    flow = FlowJax(
        dims=2,
        key=flow_key,
        device="cpu",
        data_transform=FlowTransform(
            parameters=[f"x_{i}" for i in range(2)], xp=jnp
        ),
    )

    key, samples_key = jax.random.split(key)
    x = jax.random.normal(samples_key, (100, 2))

    flow.fit_data_transform(x)

    log_prob = flow.log_prob(x)

    key_data = jax.random.key_data(flow_key)
    re_key = jax.random.wrap_key_data(key_data)
    assert flow_key == re_key

    with AspireFile(tmp_path / "result.h5", "w") as f:
        flow.save(f, "flow")

    with AspireFile(tmp_path / "result.h5", "r") as f:
        loaded_flow = FlowJax.load(f, "flow")

    # Check if the loaded flow is equivalent to the original flow
    assert loaded_flow.dims == flow.dims

    log_prob_loaded = loaded_flow.log_prob(x)
    assert jnp.allclose(log_prob, log_prob_loaded)
