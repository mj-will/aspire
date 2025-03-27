from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@dataclass
class History:
    """Base class for storing history of a sampler."""

    pass


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
        h5_file.create_dataset(
            path + "/training_loss", data=self.training_loss
        )
        h5_file.create_dataset(
            path + "/validation_loss", data=self.validation_loss
        )


@dataclass
class SMCHistory(History):
    log_norm_ratio: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    ess: list[float] = field(default_factory=list)
    ess_target: list[float] = field(default_factory=list)

    def save(self, h5_file, path="smc_history"):
        """Save the history to an HDF5 file."""
        h5_file.create_dataset(
            path + "/log_norm_ratio", data=self.log_norm_ratio
        )
        h5_file.create_dataset(path + "/beta", data=self.beta)
        h5_file.create_dataset(path + "/ess", data=self.ess)
        h5_file.create_dataset(path + "/ess_target", data=self.ess_target)

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

    def plot(self) -> Figure:
        methods = [
            self.plot_beta,
            self.plot_log_norm_ratio,
            self.plot_ess,
            self.plot_ess_target,
        ]

        fig, axs = plt.subplots(len(methods), 1, sharex=True)

        for method, ax in zip(methods, axs):
            method(ax)

        for ax in axs[:-1]:
            ax.set_xlabel("")

        return fig
