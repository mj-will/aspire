from typing import Callable

import flowjax.bijections
import flowjax.distributions
import flowjax.flows
import jax
import jax.numpy as jnp
import jax.random as jrandom


def get_flow_function_class(name: str) -> Callable:
    try:
        return getattr(flowjax.flows, name)
    except AttributeError:
        raise ValueError(f"Unknown flow function: {name}")


def get_bijection_class(name: str) -> Callable:
    try:
        return getattr(flowjax.bijections, name)
    except AttributeError:
        raise ValueError(f"Unknown bijection: {name}")


def get_flow(
    *,
    key: jax.Array,
    dims: int,
    flow_type: str | Callable = "masked_autoregressive_flow",
    bijection_type: str | flowjax.bijections.AbstractBijection | None = None,
    bijection_kwargs: dict | None = None,
    **kwargs,
) -> flowjax.distributions.Transformed:
    if isinstance(flow_type, str):
        flow_type = get_flow_function_class(flow_type)

    if isinstance(bijection_type, str):
        bijection_type = get_bijection_class(bijection_type)
    if bijection_type is not None:
        transformer = bijection_type(**bijection_kwargs)
    else:
        transformer = None

    if bijection_kwargs is None:
        bijection_kwargs = {}

    base_dist = flowjax.distributions.Normal(jnp.zeros(dims))
    key, subkey = jrandom.split(key)
    return flow_type(
        subkey,
        base_dist=base_dist,
        transformer=transformer,
        **kwargs,
    )
