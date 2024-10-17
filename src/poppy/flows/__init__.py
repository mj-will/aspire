def get_flow_wrapper(flow_matching: bool = False):
    """Get the wrapper for the flow implementation."""
    from ..backend import get_backend

    backend = get_backend()
    if backend.name == "torch":
        from .torch.flows import ZukoFlow, ZukoFlowMatching

        if flow_matching:
            return ZukoFlowMatching
        else:
            return ZukoFlow
    elif backend.name == "jax":
        from .jax.flows import FlowJax

        if flow_matching:
            raise NotImplementedError(
                "Flow matching not implemented for JAX backend"
            )
        return FlowJax
    else:
        raise ValueError(f"Unknown backend: {backend}")
