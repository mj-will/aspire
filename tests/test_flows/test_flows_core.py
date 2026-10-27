import pytest

from aspire.utils import get_flow_class


@pytest.mark.requires("zuko")
def test_get_flow_class_zuko():
    FlowClass = get_flow_class(backend="zuko", flow_matching=False)
    import array_api_compat.torch as torch_api
    from sequince_density.flows.zuko import ZukoFlow

    assert FlowClass is ZukoFlow
    assert FlowClass.xp is torch_api


@pytest.mark.requires("zuko")
def test_get_flow_class_zuko_flow_matching_not_implemented():
    with pytest.raises(NotImplementedError, match="Flow matching"):
        get_flow_class(backend="zuko", flow_matching=True)


@pytest.mark.requires("flowjax")
def test_get_flow_class_flowjax():
    FlowClass = get_flow_class(backend="flowjax", flow_matching=False)
    import jax.numpy as jnp
    from sequince_density.flows.flowjax import FlowJax

    assert FlowClass is FlowJax
    assert FlowClass.xp is jnp


@pytest.mark.requires("flowjax")
def test_get_flow_class_flowjax_flow_matching_not_implemented():
    with pytest.raises(
        NotImplementedError,
        match="Flow matching",
    ):
        get_flow_class(backend="flowjax", flow_matching=True)


def test_get_flow_class_external_backend_not_found():
    msg = "Unknown backend 'unknown_backend'. Available backends: "
    with pytest.raises(ValueError, match=msg):
        get_flow_class(backend="unknown_backend")
