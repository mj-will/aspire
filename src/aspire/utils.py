from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import array_api_compat.numpy as np
import h5py
import wrapt
from array_api_compat import (
    array_namespace,
    is_jax_array,
    is_torch_array,
    is_torch_namespace,
    to_device,
)

if TYPE_CHECKING:
    from multiprocessing import Pool

    from array_api_compat.common._typing import Array

    from .aspire import Aspire

logger = logging.getLogger(__name__)


def configure_logger(
    log_level: str | int = "INFO",
    additional_loggers: list[str] = None,
    include_aspire_loggers: bool = True,
) -> logging.Logger:
    """Configure the logger.

    Adds a stream handler to the logger.

    Parameters
    ----------
    log_level : str or int, optional
        The log level to use. Defaults to "INFO".
    additional_loggers : list of str, optional
        Additional loggers to configure. Defaults to None.
    include_aspire_loggers : bool, optional
        Whether to include all loggers that start with "aspire_" or "aspire-".
        Defaults to True.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger("aspire")
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - aspire - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    additional_loggers = additional_loggers or []
    for name in logger.manager.loggerDict:
        if include_aspire_loggers and (
            name.startswith("aspire_") or name.startswith("aspire-")
        ):
            additional_loggers.append(name)

    for name in additional_loggers:
        dep_logger = logging.getLogger(name)
        dep_logger.setLevel(log_level)
        dep_logger.handlers.clear()
        for handler in logger.handlers:
            dep_logger.addHandler(handler)
        dep_logger.propagate = False

    return logger


class PoolHandler:
    """Context manager to temporarily replace the log_likelihood method of a
    aspire instance with a version that uses a multiprocessing pool to
    parallelize computation.

    Parameters
    ----------
    aspire_instance : aspire
        The aspire instance to modify. The log_likelihood method of this
        instance must accept a :code:`map_fn` keyword argument.
    pool : multiprocessing.Pool
        The pool to use for parallel computation.
    close_pool : bool, optional
        Whether to close the pool when exiting the context manager.
        Defaults to True.
    parallelize_prior : bool, optional
        Whether to parallelize the log_prior method as well. Defaults to False.
        If True, the log_prior method of the aspire instance must also
        accept a :code:`map_fn` keyword argument.
    """

    def __init__(
        self,
        aspire_instance: Aspire,
        pool: Pool,
        close_pool: bool = True,
        parallelize_prior: bool = False,
    ):
        self.parallelize_prior = parallelize_prior
        self.aspire_instance = aspire_instance
        self.pool = pool
        self.close_pool = close_pool

    @property
    def aspire_instance(self):
        return self._aspire_instance

    @aspire_instance.setter
    def aspire_instance(self, value: Aspire):
        signature = inspect.signature(value.log_likelihood)
        if "map_fn" not in signature.parameters:
            raise ValueError(
                "The log_likelihood method of the Aspire instance must accept a"
                " 'map_fn' keyword argument."
            )
        signature = inspect.signature(value.log_prior)
        if "map_fn" not in signature.parameters and self.parallelize_prior:
            raise ValueError(
                "The log_prior method of the Aspire instance must accept a"
                " 'map_fn' keyword argument if parallelize_prior is True."
            )
        self._aspire_instance = value

    def __enter__(self):
        self.original_log_likelihood = self.aspire_instance.log_likelihood
        self.original_log_prior = self.aspire_instance.log_prior
        if self.pool is not None:
            logger.debug("Updating map function in log-likelihood method")
            self.aspire_instance.log_likelihood = partial(
                self.original_log_likelihood, map_fn=self.pool.map
            )
            if self.parallelize_prior:
                logger.debug("Updating map function in log-prior method")
                self.aspire_instance.log_prior = partial(
                    self.original_log_prior, map_fn=self.pool.map
                )
        return self.pool

    def __exit__(self, exc_type, exc_value, traceback):
        self.aspire_instance.log_likelihood = self.original_log_likelihood
        self.aspire_instance.log_prior = self.original_log_prior
        if self.close_pool:
            logger.debug("Closing pool")
            self.pool.close()
            self.pool.join()
        else:
            logger.debug("Not closing pool")


def logit(x: Array, eps: float | None = None) -> tuple[Array, Array]:
    """Logit function that also returns log Jacobian determinant.

    Parameters
    ----------
    x : float or ndarray
        Array of values
    eps : float, optional
        Epsilon value used to clamp inputs to [eps, 1 - eps]. If None, then
        inputs are not clamped.

    Returns
    -------
    float or ndarray
        Rescaled values.
    float or ndarray
        Log Jacobian determinant.
    """
    xp = array_namespace(x)
    if eps:
        x = xp.clip(x, eps, 1 - eps)
    y = xp.log(x) - xp.log1p(-x)
    log_j = (-xp.log(x) - xp.log1p(-x)).sum(-1)
    return y, log_j


def sigmoid(x: Array) -> tuple[Array, Array]:
    """Sigmoid function that also returns log Jacobian determinant.

    Parameters
    ----------
    x : float or ndarray
        Array of values

    Returns
    -------
    float or ndarray
        Rescaled values.
    float or ndarray
        Log Jacobian determinant.
    """
    xp = array_namespace(x)
    x = xp.divide(1, 1 + xp.exp(-x))
    log_j = (xp.log(x) + xp.log1p(-x)).sum(-1)
    return x, log_j


def logsumexp(x: Array, axis: int | None = None) -> Array:
    """Implementation of logsumexp that works with array api.

    This will be removed once the implementation in scipy is compatible.
    """
    xp = array_namespace(x)
    c = x.max()
    return c + xp.log(xp.sum(xp.exp(x - c), axis=axis))


def to_numpy(x: Array, **kwargs) -> np.ndarray:
    """Convert an array to a numpy array.

    This automatically moves the array to the CPU.

    Parameters
    ----------
    x : Array
        The array to convert.
    kwargs : dict
        Additional keyword arguments to pass to numpy.asarray.
    """
    try:
        return np.asarray(to_device(x, "cpu"), **kwargs)
    except (ValueError, NotImplementedError):
        return np.asarray(x, **kwargs)


def asarray(x, xp: Any = None, **kwargs) -> Array:
    """Convert an array to the specified array API.

    Parameters
    ----------
    x : Array
        The array to convert.
    xp : Any
        The array API to use for the conversion. If None, the array API
        is inferred from the input array.
    kwargs : dict
        Additional keyword arguments to pass to xp.asarray.
    """
    if is_jax_array(x) and is_torch_namespace(xp):
        return xp.utils.dlpack.from_dlpack(x)
    else:
        return xp.asarray(x, **kwargs)


def resolve_dtype(dtype: Any | str | None, xp: Any) -> Any | None:
    """Resolve a dtype specification into an XP-specific dtype.

    Parameters
    ----------
    dtype : Any | str | None
        The dtype specification. Can be None, a string, or a dtype-like object.
    xp : module
        The array API module that should interpret the dtype.

    Returns
    -------
    Any | None
        The resolved dtype object compatible with ``xp`` (or None if unspecified).
    """
    if dtype is None or xp is None:
        return dtype

    if isinstance(dtype, str):
        dtype_name = _dtype_to_name(dtype)
        if is_torch_namespace(xp):
            resolved = getattr(xp, dtype_name, None)
            if resolved is None:
                raise ValueError(
                    f"Unknown dtype '{dtype}' for namespace {xp.__name__}"
                )
            return resolved
        try:
            return xp.dtype(dtype_name)
        except (AttributeError, TypeError, ValueError):
            resolved = getattr(xp, dtype_name, None)
            if resolved is not None:
                return resolved
            raise ValueError(
                f"Unknown dtype '{dtype}' for namespace {getattr(xp, '__name__', xp)}"
            )

    if is_torch_namespace(xp):
        return dtype

    try:
        return xp.dtype(dtype)
    except (AttributeError, TypeError, ValueError):
        return dtype


def _dtype_to_name(dtype: Any | str | None) -> str | None:
    """Extract a canonical (lowercase) name for a dtype-like object."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        name = dtype
    elif hasattr(dtype, "name") and getattr(dtype, "name"):
        name = dtype.name
    elif hasattr(dtype, "__name__"):
        name = dtype.__name__
    else:
        text = str(dtype)
        if text.startswith("<class '") and text.endswith("'>"):
            text = text.split("'")[1]
        if text.startswith("dtype(") and text.endswith(")"):
            inner = text[6:-1].strip("'\" ")
            text = inner or text
        name = text
    name = name.split(".")[-1]
    return name.strip(" '\"<>").lower()


def convert_dtype(
    dtype: Any | str | None,
    target_xp: Any,
    *,
    source_xp: Any | None = None,
) -> Any | None:
    """Convert a dtype between array API namespaces.

    Parameters
    ----------
    dtype : Any | str | None
        The dtype to convert. Can be a dtype object, string, or None.
    target_xp : module
        The target array API namespace to convert the dtype into.
    source_xp : module, optional
        The source namespace of the dtype. Provided for API symmetry and future
        use; currently unused but accepted.

    Returns
    -------
    Any | None
        The dtype object compatible with ``target_xp`` (or None if ``dtype`` is None).
    """
    if dtype is None:
        return None
    if target_xp is None:
        raise ValueError("target_xp must be provided to convert dtype.")

    target_name = getattr(target_xp, "__name__", "")
    dtype_module = getattr(dtype, "__module__", "")
    if dtype_module.startswith(target_name):
        return dtype
    if is_torch_namespace(target_xp) and str(dtype).startswith("torch."):
        return dtype

    name = _dtype_to_name(dtype)
    if not name:
        raise ValueError(f"Could not infer dtype name from {dtype!r}")

    candidates = dict.fromkeys(
        [name, name.lower(), name.upper(), name.capitalize()]
    )
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return resolve_dtype(candidate, target_xp)
        except ValueError as exc:
            last_error = exc

    # Fallback to direct attribute lookup
    attr = getattr(target_xp, name, None) or getattr(
        target_xp, name.lower(), None
    )
    if attr is not None:
        return attr

    raise ValueError(
        f"Unable to convert dtype {dtype!r} to namespace {target_name}"
    ) from last_error


def copy_array(x, xp: Any = None) -> Array:
    """Copy an array based on the array API being used.

    This uses the most appropriate method to copy the array
    depending on the array API.

    Parameters
    ----------
    x : Array
        The array to copy.
    xp : Any
        The array API to use for the copy.

    Returns
    -------
    Array
        The copied array.
    """
    if xp is None:
        xp = array_namespace(x)
    # torch does not play nicely since it complains about copying tensors
    if is_torch_namespace(xp):
        if is_torch_array(x):
            return xp.clone(x)
        else:
            return xp.as_tensor(x)
    else:
        try:
            return xp.copy(x)
        except (AttributeError, TypeError):
            # Fallback for array APIs that do not have a copy method
            return xp.array(x, copy=True)


def effective_sample_size(log_w: Array) -> float:
    xp = array_namespace(log_w)
    return xp.exp(xp.asarray(logsumexp(log_w) * 2 - logsumexp(log_w * 2)))


@contextmanager
def disable_gradients(xp, inference: bool = True):
    """Disable gradients for a specific array API.

    Usage:

    ```python
    with disable_gradients(xp):
        # Do something
    ```

    Parameters
    ----------
    xp : module
        The array API module to use.
    inference : bool, optional
        When using PyTorch, set to True to enable inference mode.
    """
    if is_torch_namespace(xp):
        if inference:
            with xp.inference_mode():
                yield
        else:
            with xp.no_grad():
                yield
    else:
        yield


def encode_dtype(xp, dtype):
    """Encode a dtype for storage in an HDF5 file.

    Parameters
    ----------
    xp : module
        The array API module to use.
    dtype : dtype
        The dtype to encode.

    Returns
    -------
    str
        The encoded dtype.
    """
    if dtype is None:
        return None
    return {
        "__dtype__": True,
        "xp": xp.__name__,
        "dtype": _dtype_to_name(dtype),
    }


def decode_dtype(xp, encoded_dtype):
    """Decode a dtype from an HDF5 file.

    Parameters
    ----------
    xp : module
        The array API module to use.
    encoded_dtype : dict
        The encoded dtype.

    Returns
    -------
    dtype
        The decoded dtype.
    """
    if isinstance(encoded_dtype, dict) and encoded_dtype.get("__dtype__"):
        if encoded_dtype["xp"] != xp.__name__:
            raise ValueError(
                f"Encoded dtype xp {encoded_dtype['xp']} does not match "
                f"current xp {xp.__name__}"
            )
        if is_torch_namespace(xp):
            return getattr(xp, encoded_dtype["dtype"].split(".")[-1])
        else:
            return xp.dtype(encoded_dtype["dtype"].split(".")[-1])
    else:
        return encoded_dtype


def encode_for_hdf5(value: Any) -> Any:
    """Encode a value for storage in an HDF5 file.

    Special cases:
    - None is replaced with "__none__"
    - Empty dictionaries are replaced with "__empty_dict__"
    """
    if is_jax_array(value) or is_torch_array(value):
        return to_numpy(value)
    if isinstance(value, CallHistory):
        return value.to_dict(list_to_dict=True)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            dt = h5py.string_dtype(encoding="utf-8")
            return np.array(value, dtype=dt)
        return [encode_for_hdf5(v) for v in value]
    if isinstance(value, set):
        return {encode_for_hdf5(v) for v in value}
    if isinstance(value, dict):
        if not value:
            return "__empty_dict__"
        else:
            return {k: encode_for_hdf5(v) for k, v in value.items()}
    if value is None:
        return "__none__"

    return value


def decode_from_hdf5(value: Any) -> Any:
    """Decode a value loaded from an HDF5 file, reversing encode_for_hdf5."""
    if isinstance(value, bytes):  # HDF5 may store strings as bytes
        value = value.decode("utf-8")

    if isinstance(value, str):
        if value == "__none__":
            return None
        if value == "__empty_dict__":
            return {}

    if isinstance(value, np.ndarray):
        # Try to collapse 0-D arrays into scalars
        if value.shape == ():
            return value.item()
        if value.dtype.kind in {"S", "O"}:
            try:
                return value.astype(str).tolist()
            except Exception:
                # fallback: leave as ndarray
                return value
        return value

    if isinstance(value, list):
        return [decode_from_hdf5(v) for v in value]
    if isinstance(value, tuple):
        return tuple(decode_from_hdf5(v) for v in value)
    if isinstance(value, set):
        return {decode_from_hdf5(v) for v in value}
    if isinstance(value, dict):
        return {
            k.decode("utf-8"): decode_from_hdf5(v) for k, v in value.items()
        }

    # Fallback for ints, floats, strs, etc.
    return value


def recursively_save_to_h5_file(h5_file, path, dictionary):
    """Save a dictionary to an HDF5 file with flattened keys under a given group path."""
    # Ensure the group exists (or open it if already present)
    group = h5_file.require_group(path)

    def _save_flattened(g, prefix, d):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _save_flattened(g, full_key, value)
            else:
                try:
                    g.create_dataset(full_key, data=encode_for_hdf5(value))
                except TypeError as error:
                    try:
                        # Try saving as a string
                        dt = h5py.string_dtype(encoding="utf-8")
                        g.create_dataset(
                            full_key, data=np.array(str(value), dtype=dt)
                        )
                    except Exception:
                        raise RuntimeError(
                            f"Cannot save key {full_key} with value {value} to HDF5 file."
                        ) from error

    _save_flattened(group, "", dictionary)


def load_from_h5_file(h5_file, path):
    """Load a flattened dictionary from an HDF5 group and rebuild nesting."""
    group = h5_file[path]
    result = {}

    for key, dataset in group.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = decode_from_hdf5(dataset[()])

    return result


def get_package_version(package_name: str) -> str:
    """Get the version of a package.

    Parameters
    ----------
    package_name : str
        The name of the package.

    Returns
    -------
    str
        The version of the package.
    """
    try:
        module = __import__(package_name)
        return module.__version__
    except ImportError:
        return "not installed"


class AspireFile(h5py.File):
    """A subclass of h5py.File that adds metadata to the file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_aspire_metadata()

    def _set_aspire_metadata(self):
        from . import __version__ as aspire_version

        if self.mode in {"w", "w-", "a", "r+"}:
            self.attrs["aspire_version"] = aspire_version
        else:
            aspire_version = self.attrs.get("aspire_version", "unknown")
            if aspire_version != "unknown":
                logger.warning(
                    f"Opened Aspire file created with version {aspire_version}. "
                    f"Current version is {aspire_version}."
                )


def update_at_indices(x: Array, slc: Array, y: Array) -> Array:
    """Update an array at specific indices."

    This is a workaround for the fact that array API does not support
    advanced indexing with all backends.

    Examples
    --------
    >>> x = xp.array([[1, 2], [3, 4], [5, 6]])
    >>> update_at_indices(x, (slice(None), 0), xp.array([10, 20, 30]))
    [[10  2]
     [20  4]
     [30  6]]

    Parameters
    ----------
    x : Array
        The array to update.
    slc : Array
        The indices to update.
    y : Array
        The values to set at the indices.

    Returns
    -------
    Array
        The updated array.
    """
    try:
        x[slc] = y
        return x
    except TypeError:
        return x.at[slc].set(y)


@dataclass
class CallHistory:
    """Class to store the history of calls to a function.

    Attributes
    ----------
    args : list[tuple]
        The positional arguments of each call.
    kwargs : list[dict]
        The keyword arguments of each call.
    """

    args: list[tuple]
    kwargs: list[dict]

    def to_dict(self, list_to_dict: bool = False) -> dict[str, Any]:
        """Convert the call history to a dictionary.

        Parameters
        ----------
        list_to_dict : bool
            If True, convert the lists of args and kwargs to dictionaries
            with string keys. If False, keep them as lists. This is useful
            when encoding the history for HDF5.
        """
        if list_to_dict:
            return {
                "args": {str(i): v for i, v in enumerate(self.args)},
                "kwargs": {str(i): v for i, v in enumerate(self.kwargs)},
            }
        else:
            return {
                "args": [list(arg) for arg in self.args],
                "kwargs": [dict(kwarg) for kwarg in self.kwargs],
            }


def track_calls(wrapped=None):
    """Decorator to track calls to a function.

    The decorator adds a :code:`calls` attribute to the wrapped function,
    which is a :py:class:`CallHistory` object that stores the arguments and
    keyword arguments of each call.
    """

    @wrapt.decorator
    def wrapper(wrapped_func, instance, args, kwargs):
        # If instance is provided, we're dealing with a method.
        if instance:
            # Attach `calls` attribute to the method's `__func__`, which is the original function
            if not hasattr(wrapped_func.__func__, "calls"):
                wrapped_func.__func__.calls = CallHistory([], [])
            wrapped_func.__func__.calls.args.append(args)
            wrapped_func.__func__.calls.kwargs.append(kwargs)
        else:
            # For standalone functions, attach `calls` directly to the function
            if not hasattr(wrapped_func, "calls"):
                wrapped_func.calls = CallHistory([], [])
            wrapped_func.calls.args.append(args)
            wrapped_func.calls.kwargs.append(kwargs)

        # Call the original wrapped function
        return wrapped_func(*args, **kwargs)

    return wrapper(wrapped) if wrapped else wrapper
