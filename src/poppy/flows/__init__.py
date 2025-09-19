def get_flow_wrapper(backend: str = "zuko", flow_matching: bool = False):
    """Get the wrapper for the flow implementation."""
    if backend == "zuko":
        import array_api_compat.torch as torch_api

        from .torch.flows import ZukoFlow, ZukoFlowMatching

        if flow_matching:
            return ZukoFlowMatching, torch_api
        else:
            return ZukoFlow, torch_api
    elif backend == "flowjax":
        import jax.numpy as jnp

        from .jax.flows import FlowJax

        if flow_matching:
            raise NotImplementedError(
                "Flow matching not implemented for JAX backend"
            )
        return FlowJax, jnp
    else:
        from importlib.metadata import entry_points

        eps = {
            ep.name.lower(): ep
            for ep in entry_points().get("aspire.flows", [])
        }
        if backend in eps:
            FlowClass = eps[backend].load()
            xp = getattr(FlowClass, "xp", None)
            if xp is None:
                raise ValueError(
                    f"Flow class {backend} does not define an `xp` attribute"
                )
            return FlowClass, xp
        else:
            raise ValueError(
                f"Unknown flow class: {backend}. Available classes: {list(eps.keys())}"
            )
