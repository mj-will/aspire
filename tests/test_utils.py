import functools
import pickle

import array_api_compat.numpy as np_xp
import array_api_compat.torch as torch_xp
import h5py
import jax.numpy as jnp
import pytest

from aspire.utils import (
    configure_logger,
    convert_dtype,
    dump_state,
    function_id,
    resolve_dtype,
)


def _dtype_name(dtype):
    if hasattr(dtype, "name") and dtype.name:
        return dtype.name
    if hasattr(dtype, "__name__"):
        return dtype.__name__
    text = str(dtype)
    if text.startswith("<class '") and text.endswith("'>"):
        text = text.split("'")[1]
    if text.startswith("dtype(") and text.endswith(")"):
        text = text[6:-1].strip("'\" ")
    return text.split(".")[-1]


@pytest.mark.parametrize(
    ("source_dtype", "target_xp", "expected"),
    [
        (np_xp.dtype("float32"), torch_xp, torch_xp.float32),
        (torch_xp.float64, np_xp, np_xp.dtype("float64")),
        (jnp.float16, torch_xp, torch_xp.float16),
        ("float32", jnp, jnp.dtype("float32")),
    ],
)
def test_convert_dtype_cross_namespace(source_dtype, target_xp, expected):
    converted = convert_dtype(source_dtype, target_xp)
    assert _dtype_name(converted) == _dtype_name(expected)


def test_convert_dtype_same_namespace_returns_original():
    dtype = np_xp.dtype("float32")
    converted = convert_dtype(dtype, np_xp)
    assert converted == dtype


def test_convert_dtype_none_returns_none():
    assert convert_dtype(None, torch_xp) is None


def test_convert_dtype_invalid_raises():
    with pytest.raises(ValueError):
        convert_dtype("not_a_real_dtype", torch_xp)


@pytest.mark.parametrize(
    ("value", "xp", "expected_name"),
    [
        ("float32", np_xp, "float32"),
        ("float64", torch_xp, "float64"),
        ("float16", jnp, "float16"),
        (np_xp.dtype("int32"), np_xp, "int32"),
        (torch_xp.float32, torch_xp, "float32"),
        (jnp.dtype("float64"), jnp, "float64"),
    ],
)
def test_resolve_dtype_matches_namespace(value, xp, expected_name):
    resolved = resolve_dtype(value, xp)
    assert _dtype_name(resolved) == expected_name


def test_resolve_dtype_passes_through_torch_dtype():
    dtype = torch_xp.float32
    assert resolve_dtype(dtype, torch_xp) is dtype


def test_resolve_dtype_errors_on_unknown():
    with pytest.raises(ValueError):
        resolve_dtype("not_a_real_dtype", np_xp)


def test_dump_state_round_trip(tmp_path):
    state = {"foo": [1, 2, 3], "bar": {"nested": "ok"}}
    filename = tmp_path / "checkpoint.h5"
    with h5py.File(filename, "w") as fp:
        dump_state(state, fp, path="checkpoints", dsetname="state")
        stored = fp["checkpoints"]["state"][...]
    restored = pickle.loads(stored.tobytes())
    assert restored == state


# Define a simple function for testing
def _foo(x):
    return x


@pytest.mark.parametrize(
    "fn", [lambda x: x, functools.partial(lambda x, y: x, y=1), _foo]
)
def test_function_id(fn):
    fn_id = function_id(fn)
    assert isinstance(fn_id, str)
    # Calling again should give the same result
    assert fn_id == function_id(fn)


@pytest.mark.parametrize("log_file", [None, "test.log"])
def test_configure_logger(tmp_path, caplog, log_file):
    if log_file is not None:
        log_file = tmp_path / log_file
    logger = configure_logger(log_level="DEBUG", log_file=log_file)

    with caplog.at_level("DEBUG"):
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
    assert "This is a debug message." in caplog.text
    assert "This is an info message." in caplog.text

    if log_file is not None:
        with open(log_file, "r") as f:
            logs = f.read()
        assert "This is a debug message." in logs
        assert "This is an info message." in logs


def test_enable_scipy_array_api():
    import os

    from aspire.utils import enable_scipy_array_api

    # Clear the environment variable if it exists
    orig_val = os.environ.pop("SCIPY_ARRAY_API", None)

    enable_scipy_array_api()
    assert os.environ.get("SCIPY_ARRAY_API") == "1"
    # Restore original environment variable
    if orig_val is not None:
        os.environ["SCIPY_ARRAY_API"] = orig_val


def test_enable_scipy_array_api_respects_existing_setting():
    import os

    from aspire.utils import enable_scipy_array_api

    orig_val = os.environ.get("SCIPY_ARRAY_API", None)
    os.environ["SCIPY_ARRAY_API"] = "0"

    # Catching warnings to verify that a warning is issued
    with pytest.warns(RuntimeWarning, match="SCIPY_ARRAY_API=0"):
        enable_scipy_array_api()
    assert os.environ.get("SCIPY_ARRAY_API") == "0"
    # Reset the environment variable for other tests
    if orig_val is not None:
        os.environ["SCIPY_ARRAY_API"] = orig_val
    else:
        os.environ.pop("SCIPY_ARRAY_API", None)
