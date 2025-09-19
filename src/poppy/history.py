from __future__ import annotations

import copy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .utils import recursively_save_to_h5_file


@dataclass
class History:
    """Base class for storing history of a sampler."""

    def save(self, h5_file, path="history"):
        """Save the history to an HDF5 file."""
        dictionary = copy.deepcopy(self.__dict__)
        recursively_save_to_h5_file(h5_file, path, dictionary)


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

    def save(self, h5_file, path="smc_history"):
        """Save the history to an HDF5 file."""
        super().save(h5_file, path=path)

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
