import copy


def plot_comparison(
    *samples, parameters=None, per_samples_kwargs=None, labels=None, **kwargs
):
    """
    Plot a comparison of multiple samples.
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
        per_samples_kwargs = [{}] * len(samples)

    fig = None
    for i, sample in enumerate(samples):
        kwds = copy.deepcopy(default_kwargs)
        color = per_samples_kwargs[i].pop("color", f"C{i}")
        kwds["color"] = color
        kwds["hist_kwargs"]["color"] = color
        kwds.update(per_samples_kwargs[i])
        fig = sample.plot_corner(fig=fig, parameters=parameters, **kwds)

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
