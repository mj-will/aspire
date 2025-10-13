import pytest
import torch

from aspire.flows.torch.flows import ZukoFlow
from aspire.transforms import FlowTransform
from aspire.utils import AspireFile


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zuko_flow(dtype):
    dims = 3
    parameters = [f"x_{i}" for i in range(dims)]

    data_transform = FlowTransform(
        parameters=parameters, xp=torch, dtype=dtype
    )

    # Create an instance of ZukoFlow
    flow = ZukoFlow(
        dims=dims,
        flow_class="MAF",
        seed=42,
        device="cpu",
        data_transform=data_transform,
    )

    x = torch.randn(100, dims, device=flow.device)

    flow.fit_data_transform(x)

    # Check if the flow is initialized correctly
    assert flow.dims == dims

    # Check if the flow is an instance of ZukoFlow
    assert isinstance(flow, ZukoFlow)

    # Check if the flow has a valid flow attribute
    assert flow.flow is not None

    x = torch.tensor([0.1, 0.2, 0.3], device=flow.device)

    log_prob = flow.log_prob(x)

    assert log_prob.shape == (1,)


def test_zuko_flow_save_and_load(tmp_path):
    flow = ZukoFlow(
        dims=2,
        flow_class="MAF",
        seed=42,
        device="cpu",
    )

    x = torch.randn(100, 2, device=flow.device)

    with AspireFile(tmp_path / "result.h5", "w") as f:
        flow.save(f, "flow")

    with AspireFile(tmp_path / "result.h5", "r") as f:
        loaded_flow = ZukoFlow.load(f, "flow")

    # Check if the loaded flow is equivalent to the original flow
    assert loaded_flow.dims == flow.dims
    assert loaded_flow.flow is not None

    log_prob_original = flow.log_prob(x)
    log_prob_loaded = loaded_flow.log_prob(x)

    assert torch.allclose(log_prob_original, log_prob_loaded)
