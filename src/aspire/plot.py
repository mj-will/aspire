from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_comparison(
    *samples,
    parameters: list[str] | None = None,
    per_samples_kwargs: list[dict[str, Any]] | None = None,
    labels: list[str] | None = None,
    **kwargs,
) -> Figure:
    """
    Plot a comparison of multiple samples.

    Parameters
    ----------
    samples : Samples
        The samples to compare.
    parameters : list[str] | None
        The list of parameter names to plot. If None, the parameters will be
        inferred from the samples.
    per_samples_kwargs : list[dict] | None
        A list of dictionaries of keyword arguments to pass to each sample's
        :code:`plot_corner` method. If None, no additional keyword arguments
        will be passed. If provided, must have the same length as samples
    labels : list[str] | None
        A list of labels for the legend. If None, no legend will be shown. If
        provided, must have the same length as samples.
    kwargs : dict
        Additional keyword arguments to pass to each sample's
        :code:`plot_corner` method.
    """
    default_kwargs = dict(
        density=True,
        bins=30,
        color="C0",
        smooth=1.0,
        plot_datapoints=True,
        plot_density=False,
        hist_kwargs=dict(density=True, color="C0"),
    )
    default_kwargs.update(kwargs)

    if per_samples_kwargs is None:
        per_samples_kwargs = [{} for _ in samples]
    elif len(per_samples_kwargs) != len(samples):
        raise ValueError(
            "per_samples_kwargs must have the same length as samples"
        )

    fig = None
    for i, sample in enumerate(samples):
        kwds = copy.deepcopy(default_kwargs)
        sample_kwargs = copy.deepcopy(per_samples_kwargs[i])
        color = sample_kwargs.pop("color", f"C{i}")
        kwds["color"] = color
        kwds["hist_kwargs"]["color"] = color
        kwds.update(sample_kwargs)
        previous_fig = fig
        fig = sample.plot_corner(fig=fig, parameters=parameters, **kwds)

        # Corner seems to return a new figure so we make sure to close
        # it
        if previous_fig is not None and fig is not previous_fig:
            plt.close(previous_fig)

    if labels:
        fig.legend(
            labels=labels,
            loc="upper right",
            bbox_to_anchor=(0.9, 0.9),
            bbox_transform=fig.transFigure,
        )
    return fig


def plot_history_comparison(*histories):
    # Assert that all histories are of the same type
    if not all(isinstance(h, histories[0].__class__) for h in histories):
        raise ValueError("All histories must be of the same type")
    fig = histories[0].plot()
    for history in histories[1:]:
        fig = history.plot(fig=fig)
    return fig
