import logging
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flowjax.train import fit_to_data

from ...transforms import IdentityTransform
from ...utils import decode_dtype, encode_dtype, resolve_dtype
from ..base import Flow
from .utils import get_flow

logger = logging.getLogger(__name__)


class FlowJax(Flow):
    xp = jnp

    def __init__(
        self,
        dims: int,
        key=None,
        data_transform=None,
        dtype=None,
        **kwargs,
    ):
        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("The device argument is not used in FlowJax. ")
        resolved_dtype = (
            resolve_dtype(dtype, jnp)
            if dtype is not None
            else jnp.dtype(jnp.float32)
        )
        if data_transform is None:
            data_transform = IdentityTransform(self.xp, dtype=resolved_dtype)
        elif getattr(data_transform, "dtype", None) is None:
            data_transform.dtype = resolved_dtype
        super().__init__(dims, device=device, data_transform=data_transform)
        self.dtype = resolved_dtype
        if key is None:
            key = jrandom.key(0)
            logger.warning(
                "The key argument is None. "
                "A random key will be used for the flow. "
                "Results may not be reproducible."
            )
        self.key = key
        self.loc = None
        self.scale = None
        self.key, subkey = jrandom.split(self.key)
        self._flow = get_flow(
            key=subkey,
            dims=self.dims,
            dtype=self.dtype,
            **kwargs,
        )

    def fit(self, x, **kwargs):
        from ...history import FlowHistory

        x = jnp.asarray(x, dtype=self.dtype)
        x_prime = jnp.asarray(self.fit_data_transform(x), dtype=self.dtype)
        self.key, subkey = jrandom.split(self.key)
        self._flow, losses = fit_to_data(subkey, self._flow, x_prime, **kwargs)
        return FlowHistory(
            training_loss=list(losses["train"]),
            validation_loss=list(losses["val"]),
        )

    def forward(self, x, xp: Callable = jnp):
        x = jnp.asarray(x, dtype=self.dtype)
        x_prime, log_abs_det_jacobian = self.rescale(x)
        x_prime = jnp.asarray(x_prime, dtype=self.dtype)
        z, log_abs_det_jacobian_flow = self._flow.forward(x_prime)
        return xp.asarray(z), xp.asarray(
            log_abs_det_jacobian + log_abs_det_jacobian_flow
        )

    def inverse(self, z, xp: Callable = jnp):
        z = jnp.asarray(z, dtype=self.dtype)
        x_prime, log_abs_det_jacobian_flow = self._flow.inverse(z)
        x_prime = jnp.asarray(x_prime, dtype=self.dtype)
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(
            log_abs_det_jacobian + log_abs_det_jacobian_flow
        )

    def log_prob(self, x, xp: Callable = jnp):
        x = jnp.asarray(x, dtype=self.dtype)
        x_prime, log_abs_det_jacobian = self.rescale(x)
        x_prime = jnp.asarray(x_prime, dtype=self.dtype)
        log_prob = self._flow.log_prob(x_prime)
        return xp.asarray(log_prob + log_abs_det_jacobian)

    def sample(self, n_samples: int, xp: Callable = jnp):
        self.key, subkey = jrandom.split(self.key)
        x_prime = self._flow.sample(subkey, (n_samples,))
        x = self.inverse_rescale(x_prime)[0]
        return xp.asarray(x)

    def sample_and_log_prob(self, n_samples: int, xp: Callable = jnp):
        self.key, subkey = jrandom.split(self.key)
        x_prime = self._flow.sample(subkey, (n_samples,))
        log_prob = self._flow.log_prob(x_prime)
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(log_prob - log_abs_det_jacobian)

    def save(self, h5_file, path="flow"):
        import equinox as eqx
        from array_api_compat import numpy as np

        from ...utils import recursively_save_to_h5_file

        grp = h5_file.require_group(path)

        # ---- config ----
        config = self.config_dict().copy()
        config.pop("key", None)
        config["key_data"] = jax.random.key_data(self.key)
        dtype_value = config.get("dtype")
        if dtype_value is None:
            dtype_value = self.dtype
        else:
            dtype_value = jnp.dtype(dtype_value)
        print(dtype_value)
        config["dtype"] = encode_dtype(jnp, dtype_value)

        data_transform = config.pop("data_transform", None)
        if data_transform is not None:
            data_transform.save(grp, "data_transform")

        recursively_save_to_h5_file(grp, "config", config)

        # ---- save arrays ----
        arrays, _ = eqx.partition(self._flow, eqx.is_array)
        leaves, _ = jax.tree_util.tree_flatten(arrays)

        params_grp = grp.require_group("params")
        # clear old datasets
        for name in list(params_grp.keys()):
            del params_grp[name]

        for i, p in enumerate(leaves):
            params_grp.create_dataset(str(i), data=np.asarray(p))

    @classmethod
    def load(cls, h5_file, path="flow"):
        import equinox as eqx

        from ...utils import load_from_h5_file

        grp = h5_file[path]

        # ---- config ----
        config = load_from_h5_file(grp, "config")
        config["dtype"] = decode_dtype(jnp, config.get("dtype"))
        if "data_transform" in grp:
            from ...transforms import BaseTransform

            config["data_transform"] = BaseTransform.load(
                grp,
                "data_transform",
                strict=False,
            )

        key_data = config.pop("key_data", None)
        if key_data is not None:
            config["key"] = jax.random.wrap_key_data(key_data)

        kwargs = config.pop("kwargs", {})
        config.update(kwargs)

        # build object (will replace its _flow)
        obj = cls(**config)

        # ---- load arrays ----
        params_grp = grp["params"]
        loaded_params = [
            jnp.array(params_grp[str(i)][:]) for i in range(len(params_grp))
        ]

        # rebuild template flow
        kwargs.pop("device")
        flow_template = get_flow(key=jrandom.key(0), dims=obj.dims, **kwargs)
        arrays_template, static = eqx.partition(flow_template, eqx.is_array)

        # use treedef from template
        treedef = jax.tree_util.tree_structure(arrays_template)
        arrays = jax.tree_util.tree_unflatten(treedef, loaded_params)

        # recombine
        obj._flow = eqx.combine(static, arrays)

        return obj
