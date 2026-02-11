import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from aspire.history import History, SMCHistory
from aspire.samples import SMCSamples


def test_history_save_load(tmp_path):
    history = History()
    history.stat = [1, 2, 3]
    with h5py.File(tmp_path / "history.h5", "w") as f:
        history.save(f, path="history")

    with h5py.File(tmp_path / "history.h5", "r") as f:
        loaded_history = History.load(f, path="history")

    assert np.array_equal(loaded_history.stat, [1, 2, 3])


def test_smc_history_save_load(tmp_path):
    history = SMCHistory()
    samples = SMCSamples(
        x=np.array([[1, 2], [3, 4]]), beta=0.5, parameters=["x1", "x2"]
    )
    history.sample_history.append(samples)

    with h5py.File(tmp_path / "smc_history.h5", "w") as f:
        history.save(f, path="smc_history")

    with h5py.File(tmp_path / "smc_history.h5", "r") as f:
        loaded_history = SMCHistory.load(f, path="smc_history")

    assert isinstance(loaded_history.sample_history[0], SMCSamples)
    assert len(loaded_history.sample_history) == 1
    assert np.array_equal(loaded_history.sample_history[0].x, [[1, 2], [3, 4]])
    assert loaded_history.sample_history[0].beta == 0.5
    assert loaded_history.sample_history[0].parameters == ["x1", "x2"]


@pytest.mark.parametrize("log_w", [None, np.array([-0.5, -1.0])])
@pytest.mark.parametrize("parameters", [None, ["x1"]])
@pytest.mark.parametrize("n_samples", [None, 1])
def test_smc_history_samples(xp, log_w, parameters, n_samples):
    history = SMCHistory()
    samples1 = SMCSamples(
        x=[[1, 2], [3, 4]], xp=xp, beta=0.5, parameters=["x1", "x2"]
    )
    samples2 = SMCSamples(
        x=[[5, 6], [7, 8]], xp=xp, beta=0.6, parameters=["x1", "x2"]
    )

    if log_w is not None:
        samples1.log_w = samples1.array_to_namespace(log_w)
        samples2.log_w = samples2.array_to_namespace(log_w)

    history.sample_history.append(samples1)
    history.sample_history.append(samples2)

    assert len(history.sample_history) == 2
    assert np.array_equal(history.sample_history[0].x, [[1, 2], [3, 4]])
    assert np.array_equal(history.sample_history[1].x, [[5, 6], [7, 8]])

    fig = history.plot_sample_history(
        parameters=parameters, n_samples=n_samples
    )
    assert fig is not None
    plt.close(fig)


def test_smc_history_samples_x_axis_options():
    history = SMCHistory()
    samples1 = SMCSamples(
        x=np.array([[1, 2], [3, 4]]), beta=0.5, parameters=["x1", "x2"]
    )
    samples2 = SMCSamples(
        x=np.array([[5, 6], [7, 8]]), beta=0.6, parameters=["x1", "x2"]
    )

    for samples in (samples1, samples2):
        samples.log_likelihood = samples.array_to_namespace(
            np.array([1.0, 2.0])
        )
        samples.log_prior = samples.array_to_namespace(np.array([0.3, 0.4]))
        samples.log_q = samples.array_to_namespace(np.array([-0.2, -0.1]))

    history.sample_history.extend([samples1, samples2])

    fig_ll = history.plot_sample_history(x_axis="log_likelihood")
    assert fig_ll.axes[-1].get_xlabel() == "log likelihood"
    plt.close(fig_ll)

    fig_pt = history.plot_sample_history(x_axis="log_p_t")
    assert fig_pt.axes[-1].get_xlabel() == "log p_t(beta)"
    plt.close(fig_pt)


def test_smc_history_samples_x_axis_invalid():
    history = SMCHistory()
    samples = SMCSamples(
        x=np.array([[1, 2], [3, 4]]), beta=0.5, parameters=["x1", "x2"]
    )
    history.sample_history.append(samples)
    with pytest.raises(ValueError, match="Unsupported x_axis"):
        history.plot_sample_history(x_axis="log_w")


def test_smc_history_plot_quantile_bands():
    history = SMCHistory()
    samples1 = SMCSamples(
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        beta=0.2,
        parameters=["x1", "x2"],
    )
    samples2 = SMCSamples(
        x=np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
        beta=0.5,
        parameters=["x1", "x2"],
    )
    history.sample_history.extend([samples1, samples2])

    fig = history.plot_quantile_bands()
    assert fig is not None
    assert fig.axes[-1].get_xlabel() == "Iteration"
    assert fig.axes[0].get_ylabel() == "x1"
    assert fig.axes[1].get_ylabel() == "x2"
    plt.close(fig)


def test_smc_history_plot_quantile_bands_invalid_interval():
    history = SMCHistory()
    samples = SMCSamples(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        beta=0.2,
        parameters=["x1", "x2"],
    )
    history.sample_history.append(samples)
    with pytest.raises(ValueError, match="quantile_interval"):
        history.plot_quantile_bands(quantile_interval=(0.6, 0.9))
