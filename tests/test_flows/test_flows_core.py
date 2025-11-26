import pytest

from aspire.flows import get_flow_wrapper


def test_get_flow_wrapper_zuko():
    FlowClass, xp = get_flow_wrapper(backend="zuko", flow_matching=False)
    import array_api_compat.torch as torch_api

    from aspire.flows.torch.flows import ZukoFlow

    assert FlowClass is ZukoFlow
    assert xp is torch_api


def test_get_flow_wrapper_zuko_flow_matching():
    FlowClass, xp = get_flow_wrapper(backend="zuko", flow_matching=True)
    import array_api_compat.torch as torch_api

    from aspire.flows.torch.flows import ZukoFlowMatching

    assert FlowClass is ZukoFlowMatching
    assert xp is torch_api


def test_get_flow_wrapper_flowjax():
    FlowClass, xp = get_flow_wrapper(backend="flowjax", flow_matching=False)
    import jax.numpy as jnp

    from aspire.flows.jax.flows import FlowJax

    assert FlowClass is FlowJax
    assert xp is jnp


def test_get_flow_wrapper_flowjax_flow_matching_not_implemented():
    with pytest.raises(
        NotImplementedError,
        match="Flow matching not implemented for JAX backend",
    ):
        get_flow_wrapper(backend="flowjax", flow_matching=True)


def test_get_flow_wrapper_external_backend_not_found():
    msg = "Unknown backend 'unknown_backend'. Available backends: "
    with pytest.raises(ValueError, match=msg):
        get_flow_wrapper(backend="unknown_backend")
