import math

import pytest

from aspire import transforms
from aspire.utils import AspireFile, copy_array, update_at_indices


def _make_array(xp, data, dtype):
    if dtype is None:
        return xp.asarray(data)
    return xp.asarray(data, dtype=dtype)


def save_and_load(tmp_path, transform):
    # Save the transform to an HDF5 file
    with AspireFile(tmp_path / "result.h5", "w") as h5_file:
        transform.save(h5_file, path="flow/data_transform")

    # Load the transform from the HDF5 file
    with AspireFile(tmp_path / "result.h5", "r") as h5_file:
        loaded_transform = transform.__class__.load(
            h5_file, path="flow/data_transform"
        )
    return loaded_transform


def test_save_and_load_identity_transform(tmp_path, xp, dtype):
    data_transform = transforms.IdentityTransform(xp=xp, dtype=dtype)
    loaded_transform = save_and_load(tmp_path, data_transform)

    # Check that the loaded transform has the same parameters
    assert type(loaded_transform) is type(data_transform)
    assert loaded_transform.dtype == data_transform.dtype


def test_identity_transform_forward_inverse_roundtrip(xp, dtype):
    transform = transforms.IdentityTransform(xp=xp, dtype=dtype)
    x = _make_array(xp, [[0.0, 1.0], [1.5, -2.5]], dtype)

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    assert xp.allclose(y, x)
    assert xp.allclose(x_inv, x)
    assert xp.allclose(log_j, xp.zeros(len(x)))
    assert xp.allclose(inv_log_j, xp.zeros(len(x)))


def test_save_and_load_periodic_transform(tmp_path, xp, dtype):
    data_transform = transforms.PeriodicTransform(
        lower=0, upper=xp.pi, xp=xp, dtype=dtype
    )
    loaded_transform = save_and_load(tmp_path, data_transform)

    # Check that the loaded transform has the same parameters
    assert type(loaded_transform) is type(data_transform)
    assert loaded_transform.dtype == data_transform.dtype


def test_periodic_transform_wraps_and_inverts(xp, dtype):
    lower = 0.0
    upper = 2 * math.pi
    transform = transforms.PeriodicTransform(
        lower=lower, upper=upper, xp=xp, dtype=dtype
    )
    width = upper - lower
    raw = [-math.pi, 3 * math.pi]
    x = _make_array(xp, [[raw[0]], [raw[1]]], dtype)
    expected = [((value - lower) % width) + lower for value in raw]
    expected_arr = _make_array(xp, [[v] for v in expected], transform.dtype)

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    assert xp.allclose(y, expected_arr, atol=1e-6)
    assert xp.allclose(x_inv, expected_arr, atol=1e-6)
    assert xp.allclose(log_j, xp.zeros(y.shape[0]))
    assert xp.allclose(inv_log_j, xp.zeros(y.shape[0]))


def test_periodic_transform_multidimensional(xp, dtype):
    lower = _make_array(xp, [0.0, -math.pi], dtype)
    upper = _make_array(xp, [2 * math.pi, math.pi], dtype)
    transform = transforms.PeriodicTransform(
        lower=lower, upper=upper, xp=xp, dtype=dtype
    )
    x = _make_array(
        xp,
        [
            [-3.0, -3.5],
            [7.0, 3.2],
        ],
        dtype,
    )

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    assert y.shape == x.shape
    assert xp.allclose(x_inv, y, atol=1e-6)
    assert log_j.shape == (x.shape[0],)
    assert inv_log_j.shape == (x.shape[0],)


def test_save_and_load_affine_transform(tmp_path, rng, xp):
    dims = 3

    x = xp.asarray(rng.normal(size=(100, dims)))
    data_transform = transforms.AffineTransform(xp)
    data_transform.fit(x)

    loaded_transform = save_and_load(tmp_path, data_transform)

    # Check that the loaded transform has the same parameters
    assert loaded_transform._mean.shape == data_transform._mean.shape
    assert loaded_transform._std.shape == data_transform._std.shape
    # Check types are the same
    assert type(loaded_transform._mean) is type(data_transform._mean)
    assert type(loaded_transform._std) is type(data_transform._std)
    # Check values are close
    assert xp.allclose(loaded_transform._mean, data_transform._mean)
    assert xp.allclose(loaded_transform._std, data_transform._std)


def test_logit_transform_forward_inverse_roundtrip(xp, dtype):
    transform = transforms.LogitTransform(
        lower=-1.0, upper=2.0, xp=xp, dtype=dtype
    )
    x = _make_array(
        xp,
        [
            [-0.75, 0.0, 1.25],
            [1.5, -0.25, 0.8],
        ],
        dtype,
    )

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    assert xp.allclose(x_inv, x, atol=1e-5)
    assert log_j.shape == (x.shape[0],)
    assert inv_log_j.shape == (x.shape[0],)


def test_logit_transform_multidimensional(xp, dtype):
    lower = _make_array(xp, [-1.0, 0.0], dtype)
    upper = _make_array(xp, [1.0, 2.0], dtype)
    transform = transforms.LogitTransform(
        lower=lower, upper=upper, xp=xp, dtype=dtype
    )
    x = _make_array(
        xp,
        [
            [-0.5, 0.5],
            [0.75, 1.75],
        ],
        dtype,
    )

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    assert xp.allclose(x_inv, x, atol=1e-5)
    assert log_j.shape == (x.shape[0],)
    assert inv_log_j.shape == (x.shape[0],)


def test_save_and_load_logit_transform(tmp_path, xp, dtype):
    data_transform = transforms.LogitTransform(
        lower=-2, upper=3, xp=xp, dtype=dtype
    )
    loaded_transform = save_and_load(tmp_path, data_transform)

    # Check that the loaded transform has the same parameters
    assert loaded_transform.lower == data_transform.lower
    assert loaded_transform.upper == data_transform.upper
    assert loaded_transform.eps == data_transform.eps
    assert loaded_transform.dtype == data_transform.dtype


def test_save_and_load_probit_transform(tmp_path, xp, dtype):
    data_transform = transforms.ProbitTransform(
        lower=-2, upper=3, xp=xp, dtype=dtype
    )
    loaded_transform = save_and_load(tmp_path, data_transform)

    # Check that the loaded transform has the same parameters
    assert loaded_transform.lower == data_transform.lower
    assert loaded_transform.upper == data_transform.upper
    assert loaded_transform.eps == data_transform.eps
    assert loaded_transform.dtype == data_transform.dtype


def test_save_and_load_composite_transform(tmp_path, rng, xp, dtype):
    dims = 3

    parameters = [f"x_{i}" for i in range(dims)]
    x = xp.asarray(rng.normal(size=(100, dims)))

    transform = transforms.CompositeTransform(
        parameters=parameters,
        periodic_parameters=["x_0"],
        prior_bounds={p: [-3, 3] for p in parameters},
        xp=xp,
        dtype=dtype,
    )
    transform.fit(x)

    loaded_transform = save_and_load(tmp_path, transform)

    # Check that the loaded transform has the same parameters
    assert type(loaded_transform) is type(transform)
    assert loaded_transform.dtype == transform.dtype
    assert loaded_transform.parameters == transform.parameters
    assert (
        loaded_transform.periodic_parameters == transform.periodic_parameters
    )

    x_forward, _ = transform.forward(x)
    x_inverse, _ = transform.inverse(x_forward)

    x_forward_loaded, _ = loaded_transform.forward(x)
    x_inverse_loaded, _ = loaded_transform.inverse(x_forward_loaded)

    # Check that the forward and inverse transforms are the same
    assert xp.allclose(x_forward, x_forward_loaded)
    assert xp.allclose(x_inverse, x_inverse_loaded)


def test_composite_transform_forward_inverse_roundtrip(xp, dtype):
    parameters = ["x0", "x1", "x2"]
    prior_bounds = {p: (-3.0, 3.0) for p in parameters}
    transform = transforms.CompositeTransform(
        parameters=parameters,
        periodic_parameters=["x0"],
        prior_bounds=prior_bounds,
        xp=xp,
        dtype=dtype,
        affine_transform=True,
    )

    assert transform._periodic_transform.dtype == transform.dtype
    assert transform._bounded_transform.dtype == transform.dtype
    assert transform._affine_transform.dtype == transform.dtype

    x_fit = _make_array(
        xp,
        [
            [-2.0, -2.5, 0.0],
            [2.0, 1.5, -1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype,
    )
    transform.fit(x_fit)

    x = _make_array(
        xp,
        [
            [-4.0, -2.5, 0.0],
            [7.0, 1.5, -1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype,
    )

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    x_exp = copy_array(x)
    print(x_exp)
    print((x_exp[:, 0] + 3) % 6 - 3)
    x_exp = update_at_indices(
        x_exp,
        (slice(None), 0),
        ((x_exp[:, 0] + 3) % 6) - 3,
    )  # Wrap x0 to [-3, 3]
    print(x_exp)

    assert x.shape == y.shape
    assert xp.allclose(x_inv, x_exp, atol=1e-5)
    assert log_j.shape == (x.shape[0],)
    assert inv_log_j.shape == (x.shape[0],)


def test_composite_transform_requires_prior_bounds_for_periodic(xp, dtype):
    with pytest.raises(ValueError):
        transforms.CompositeTransform(
            parameters=[0],
            periodic_parameters=[0],
            prior_bounds=None,
            xp=xp,
            dtype=dtype,
        )


def test_composite_transform_new_instance_copies_configuration(xp, dtype):
    parameters = ["x0", "x1"]
    prior_bounds = {p: (-1.0, 1.0) for p in parameters}
    transform = transforms.CompositeTransform(
        parameters=parameters,
        periodic_parameters=[],
        prior_bounds=prior_bounds,
        xp=xp,
        dtype=dtype,
        affine_transform=False,
    )

    new_transform = transform.new_instance()
    assert new_transform.xp is transform.xp
    assert new_transform.dtype == transform.dtype
    assert new_transform.parameters == transform.parameters
    assert new_transform.periodic_parameters == transform.periodic_parameters


def test_flow_transform_config_drops_periodic_parameters(xp, dtype):
    parameters = ["a", "b"]
    prior_bounds = {p: (-2.0, 2.0) for p in parameters}
    transform = transforms.FlowTransform(
        parameters=parameters,
        prior_bounds=prior_bounds,
        xp=xp,
        dtype=dtype,
    )

    cfg = transform.config_dict()
    assert "periodic_parameters" not in cfg
    assert cfg["parameters"] == parameters


def test_logit_transform_interval_check_raises(xp):
    with pytest.raises(ValueError):
        transforms.LogitTransform(lower=0.0, upper=0.0, xp=xp, dtype=None)


def test_probit_transform_multidimensional_roundtrip(xp, dtype):
    lower = _make_array(xp, [-2.0, -1.0], dtype)
    upper = _make_array(xp, [2.0, 3.0], dtype)
    transform = transforms.ProbitTransform(
        lower=lower, upper=upper, xp=xp, dtype=dtype
    )
    x = _make_array(
        xp,
        [
            [-1.0, -0.5],
            [1.0, 2.5],
        ],
        dtype,
    )

    y, log_j = transform.forward(x)
    x_inv, inv_log_j = transform.inverse(y)

    assert xp.allclose(x_inv, x, atol=1e-5)
    assert log_j.shape == (x.shape[0],)
    assert inv_log_j.shape == (x.shape[0],)
