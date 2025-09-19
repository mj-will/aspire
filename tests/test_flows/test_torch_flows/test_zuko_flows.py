import torch
from aspire.flows.torch.flows import ZukoFlow
from aspire.transforms import FlowTransform


def test_zuko_flow():
    dims = 3
    parameters = [f"x_{i}" for i in range(dims)]

    data_transform = FlowTransform(parameters=parameters, xp=torch)

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
