import array_api_compat.numpy as np_xp
import array_api_compat.torch as torch_xp
import jax.numpy as jnp
import pytest

from aspire.utils import convert_dtype, resolve_dtype


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
