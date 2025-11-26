import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_flow_wrapper(
    backend: str = "zuko", flow_matching: bool = False
) -> tuple[type, Any]:
    """Get the wrapper for the flow implementation.

    Parameters
    ----------
    backend : str
        The backend to use. Options are "zuko" (PyTorch), "flowjax" (JAX), or
        any other registered flow class via entry points. Default is "zuko".
    flow_matching : bool, optional
        Whether to use flow matching variant of the flow. Default is False.

    Returns
    -------
    FlowClass : type
        The flow class corresponding to the specified backend.
    xp : module
        The array API module corresponding to the specified backend.
    """
    from importlib.metadata import entry_points

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
        if flow_matching:
            logger.warning(
                "Flow matching option is ignored for external backends."
            )
        eps = {
            ep.name.lower(): ep for ep in entry_points(group="aspire.flows")
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
            known_backends = ["zuko", "flowjax"] + list(eps.keys())
            raise ValueError(
                f"Unknown backend '{backend}'. Available backends: {known_backends}"
            )
