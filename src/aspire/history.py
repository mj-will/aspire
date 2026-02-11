from __future__ import annotations

import copy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .samples import SMCSamples
from .utils import load_from_h5_file, recursively_save_to_h5_file


@dataclass
class History:
    """Base class for storing history of a sampler."""

    def save(self, h5_file, path="history"):
        """Save the history to an HDF5 file."""
        dictionary = copy.deepcopy(self.__dict__)
        recursively_save_to_h5_file(h5_file, path, dictionary)

    @classmethod
    def load(cls, h5_file, path="history"):
        """Load the history from an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The open HDF5 file to load from.
        path : str, optional
            The path within the HDF5 file to load the history from.
            Default is "history".
        """
        dictionary = load_from_h5_file(h5_file, path)
        # Dataclass may have fields not present in the init signature, so we
        # filter the loaded dictionary to only include fields that are defined
        # in the dataclass
        field_names = {
            field.name for field in cls.__dataclass_fields__.values()
        }
        filtered_dict = {
            k: v for k, v in dictionary.items() if k in field_names
        }
        instance = cls(**filtered_dict)
        for k, v in dictionary.items():
            if k not in field_names:
                setattr(instance, k, v)
        return instance


@dataclass
class FlowHistory(History):
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    def plot_loss(self) -> Figure:
        """Plot the training and validation loss."""
        fig = plt.figure()
        plt.plot(self.training_loss, label="Training loss")
        plt.plot(self.validation_loss, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        return fig

    def save(self, h5_file, path="flow_history"):
        """Save the history to an HDF5 file."""
        super().save(h5_file, path=path)


@dataclass
class SMCHistory(History):
    log_norm_ratio: list[float] = field(default_factory=list)
    log_norm_ratio_var: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    ess: list[float] = field(default_factory=list)
    ess_target: list[float] = field(default_factory=list)
    eff_target: list[float] = field(default_factory=list)
    mcmc_autocorr: list[float] = field(default_factory=list)
    mcmc_acceptance: list[float] = field(default_factory=list)
    sample_history: list[SMCSamples] = field(default_factory=list)

    def save(self, h5_file, path="smc_history"):
        """Save the history to an HDF5 file.

        The sample history is saved as a separate group under the main history
        group, with one subgroup per iteration. The number of iterations is
        stored in the main history group to allow for loading the sample history
        correctly.

        Parameters
        ----------
        h5_file : h5py.File
            The open HDF5 file to save to.
        path : str, optional
            The path within the HDF5 file to save the history. Default is
            "smc_history".
        """
        dictionary = copy.deepcopy(self.__dict__)
        sample_history = dictionary.pop("sample_history", [])
        dictionary["__len_sample_history"] = len(sample_history)
        recursively_save_to_h5_file(h5_file, path, dictionary)
        for i, samples in enumerate(sample_history):
            samples.save(h5_file, path=f"{path}__sample_history/{i}")

    @classmethod
    def load(cls, h5_file, path="smc_history"):
        """Load the history from an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The open HDF5 file to load from.
        path : str, optional
            The path within the HDF5 file to load the history from. Default is
            "smc_history".

        Returns
        -------
        SMCHistory
            The loaded history instance.
        """
        dictionary = load_from_h5_file(h5_file, path)
        n_samples = int(dictionary.pop("__len_sample_history", 0))
        dictionary["sample_history"] = [
            SMCSamples.load(h5_file, path=f"{path}__sample_history/{i}")
            for i in range(n_samples)
        ]
        # Dataclass may have fields not present in the init signature, so we
        # filter the loaded dictionary to only include fields that are defined
        # in the dataclass
        field_names = {
            field.name for field in cls.__dataclass_fields__.values()
        }
        filtered_dict = {
            k: v for k, v in dictionary.items() if k in field_names
        }
        instance = cls(**filtered_dict)
        # Set any additional attributes that were not part of the dataclass
        # fields
        for k, v in dictionary.items():
            if k not in field_names:
                setattr(instance, k, v)
        return instance

    def plot_beta(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.beta)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\beta$")
        return fig

    def plot_log_norm_ratio(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.log_norm_ratio)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log evidence ratio")
        return fig

    def plot_ess(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.ess)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ESS")
        return fig

    def plot_ess_target(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.ess_target)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ESS target")
        return fig

    def plot_eff_target(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.eff_target)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Efficiency target")
        return fig

    def plot_mcmc_acceptance(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.mcmc_acceptance)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MCMC Acceptance")
        return fig

    def plot_mcmc_autocorr(self, ax=None) -> Figure | None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(self.mcmc_autocorr)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MCMC Autocorr")
        return fig

    def plot(self, fig: Figure | None = None) -> Figure:
        methods = [
            self.plot_beta,
            self.plot_log_norm_ratio,
            self.plot_ess,
            self.plot_ess_target,
            self.plot_eff_target,
            self.plot_mcmc_acceptance,
        ]

        if fig is None:
            fig, axs = plt.subplots(len(methods), 1, sharex=True)
        else:
            axs = fig.axes

        for method, ax in zip(methods, axs):
            method(ax)

        for ax in axs[:-1]:
            ax.set_xlabel("")

        return fig

    def plot_sample_history(
        self,
        n_samples=None,
        parameters=None,
        ax=None,
        cmap: str = "viridis",
        scatter_kwargs=None,
        x_axis: str = "log_p_t",
    ) -> Figure | None:
        """Plot the history of samples in the SMC sampler.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to plot from each iteration. If None, plot all samples.
        parameters : list of str, optional
            List of parameter names to plot. If None, plot all parameters.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        cmap : str, optional
            Colormap to use for plotting the samples. Default is "viridis".
        scatter_kwargs : dict, optional
            Keyword arguments to pass to the scatter function. If None, no additional arguments will be passed
        x_axis : str, optional
            Quantity to use for the x-axis. Supported values are
            :code:`"log_p_t"` and :code:`"log_likelihood"`.
            Falls back to iteration index if required fields are missing.
        """
        import numpy as np

        if x_axis not in {"log_p_t", "log_likelihood"}:
            raise ValueError(
                f"Unsupported x_axis '{x_axis}'. Expected 'log_p_t' or 'log_likelihood'."
            )

        n_parameters = (
            len(parameters)
            if parameters is not None
            else self.sample_history[0].dims
        )
        if ax is None:
            fig, ax = plt.subplots(n_parameters, 1, sharex=True)
            ax = np.atleast_1d(ax)
        else:
            fig = None

        cmap = plt.get_cmap(cmap)
        colors = cmap(np.linspace(0, 1, len(self.sample_history)))

        has_log_pt = all(
            getattr(samples, "beta", None) is not None
            and samples.log_likelihood is not None
            and samples.log_prior is not None
            and samples.log_q is not None
            for samples in self.sample_history
        )
        has_log_likelihood = all(
            samples.log_likelihood is not None
            for samples in self.sample_history
        )

        scatter_kwargs = scatter_kwargs or {}

        default_scatter_kwargs = dict(s=10)
        scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}

        for it, samples, color in zip(
            range(len(self.sample_history)), self.sample_history, colors
        ):
            samples = samples.to_numpy()
            if n_samples is not None:
                samples = samples[:n_samples]
            if parameters is not None:
                idx = [samples.parameters.index(p) for p in parameters]
                x = samples.x[:, idx]
            else:
                x = samples.x
            for i in range(x.shape[1]):
                if x_axis == "log_p_t" and has_log_pt:
                    x_axis_values = samples.log_p_t(samples.beta)
                elif x_axis == "log_likelihood" and has_log_likelihood:
                    x_axis_values = samples.log_likelihood
                else:
                    x_axis_values = it * np.ones(samples.x.shape[0])
                ax[i].scatter(
                    x_axis_values, x[:, i], color=color, **scatter_kwargs
                )

        parameters = parameters or samples.parameters
        if parameters is None:
            parameters = [f"x_{i}" for i in range(samples.x.shape[1])]
        for i, p in enumerate(parameters):
            ax[i].set_ylabel(p)

        if x_axis == "log_p_t" and has_log_pt:
            ax[-1].set_xlabel("log p_t(beta)")
        elif x_axis == "log_likelihood" and has_log_likelihood:
            ax[-1].set_xlabel("log likelihood")
        else:
            ax[-1].set_xlabel("Iteration")
        return fig

    def plot_quantile_bands(
        self,
        parameters: list[str] | None = None,
        quantile_interval: tuple[float, float] = (0.1, 0.9),
        ax=None,
        line_kwargs=None,
        band_kwargs=None,
    ) -> Figure | None:
        """Plot per-parameter quantile bands vs iteration.

        Parameters
        ----------
        parameters : list[str] | None, optional
            Parameters to plot. If None, all parameters are used.
        quantile_interval : tuple[float, float], optional
            Lower/upper quantiles to plot as a band.
        ax : matplotlib.axes.Axes | list[matplotlib.axes.Axes] | None, optional
            Axes to draw on. If None, creates a new figure.
        line_kwargs : dict | None, optional
            Keyword arguments for the median line.
        band_kwargs : dict | None, optional
            Keyword arguments for the quantile band fill.
        """
        import numpy as np

        if not self.sample_history:
            raise ValueError("No sample history available to plot.")

        q_low, q_high = quantile_interval
        if not (0.0 <= q_low < 0.5 and 0.5 < q_high <= 1.0 and q_low < q_high):
            raise ValueError(
                "quantile_interval must satisfy 0 <= low < 0.5 < high <= 1."
            )

        first = self.sample_history[0]
        all_parameters = first.parameters or [
            f"x_{i}" for i in range(first.dims)
        ]
        if parameters is None:
            parameters = all_parameters

        indices = [all_parameters.index(p) for p in parameters]
        n_params = len(indices)

        if ax is None:
            fig, axs = plt.subplots(n_params, 1, sharex=True)
            axs = np.atleast_1d(axs)
        else:
            fig = None
            axs = np.atleast_1d(ax)
            if len(axs) != n_params:
                raise ValueError(
                    "Number of axes must match number of requested parameters."
                )

        line_kwargs = {"color": "C0", "lw": 1.5, **(line_kwargs or {})}
        band_kwargs = {"color": "C0", "alpha": 0.2, **(band_kwargs or {})}

        iterations = np.arange(len(self.sample_history))
        medians = np.empty((len(self.sample_history), n_params))
        lowers = np.empty((len(self.sample_history), n_params))
        uppers = np.empty((len(self.sample_history), n_params))

        for it, samples in enumerate(self.sample_history):
            x_np = samples.to_numpy().x
            for j, idx in enumerate(indices):
                values = x_np[:, idx]
                medians[it, j] = np.quantile(values, 0.5)
                lowers[it, j] = np.quantile(values, q_low)
                uppers[it, j] = np.quantile(values, q_high)

        for j, (axis, param) in enumerate(zip(axs, parameters)):
            axis.plot(iterations, medians[:, j], **line_kwargs)
            axis.fill_between(
                iterations, lowers[:, j], uppers[:, j], **band_kwargs
            )
            axis.set_ylabel(param)

        axs[-1].set_xlabel("Iteration")
        return fig
