import matplotlib.pyplot as plt
import pytest

from aspire.plot import plot_comparison


class DummySample:
    def __init__(self):
        self.calls = []
        self._fig = object()

    def plot_corner(self, fig=None, parameters=None, **kwargs):
        self.calls.append(
            {"fig": fig, "parameters": parameters, "kwargs": kwargs}
        )
        return fig if fig is not None else self._fig


def test_plot_comparison_does_not_mutate_per_sample_kwargs():
    s0 = DummySample()
    s1 = DummySample()
    per = [{"color": "red", "bins": 10}, {"bins": 20}]

    plot_comparison(s0, s1, per_samples_kwargs=per, show_progress=True)

    assert per == [{"color": "red", "bins": 10}, {"bins": 20}]
    assert s0.calls[0]["kwargs"]["color"] == "red"
    assert s0.calls[0]["kwargs"]["hist_kwargs"]["color"] == "red"
    assert s1.calls[0]["kwargs"]["color"] == "C1"
    assert s1.calls[0]["kwargs"]["hist_kwargs"]["color"] == "C1"


def test_plot_comparison_per_samples_kwargs_length_check():
    s0 = DummySample()
    s1 = DummySample()
    with pytest.raises(ValueError, match="same length as samples"):
        plot_comparison(s0, s1, per_samples_kwargs=[{}])


def test_plot_comparison_closes_superseded_figures(monkeypatch):
    closed = []
    monkeypatch.setattr(plt, "close", lambda fig: closed.append(fig))

    class NewFigureEveryCallSample:
        def __init__(self):
            self.counter = 0

        def plot_corner(self, fig=None, parameters=None, **kwargs):
            _ = (fig, parameters, kwargs)
            self.counter += 1
            return object()

    s0 = NewFigureEveryCallSample()
    s1 = NewFigureEveryCallSample()
    s2 = NewFigureEveryCallSample()

    plot_comparison(s0, s1, s2, show_progress=True)

    # With 3 calls returning fresh figures, the first two are superseded.
    assert len(closed) == 2
