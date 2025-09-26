from aspire import transforms
from aspire.utils import AspireFile


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


def test_save_and_load_periodic_transform(tmp_path, xp, dtype):
    data_transform = transforms.PeriodicTransform(
        lower=0, upper=xp.pi, xp=xp, dtype=dtype
    )
    loaded_transform = save_and_load(tmp_path, data_transform)

    # Check that the loaded transform has the same parameters
    assert type(loaded_transform) is type(data_transform)
    assert loaded_transform.dtype == data_transform.dtype


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
