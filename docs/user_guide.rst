User Guide
==========

This guide walks through the main concepts you will use when combining
existing samples with new inference runs. It focuses on the high-level Python
API exposed by :class:`aspire.Aspire`.

Workflow overview
-----------------

1. **Describe your problem** via ``log_likelihood`` and ``log_prior``
   callables that accept :class:`aspire.samples.Samples` or compatible
   objects.
2. **Package initial draws** with :class:`aspire.samples.Samples` (or
   :class:`aspire.samples.BaseSamples`) to benefit from consistent typing,
   plotting helpers, and device-aware conversions.
3. **Fit a proposal** with :meth:`aspire.Aspire.fit` to learn a proposal
   tailored to the current posterior (normalising flow by default).
4. **Sample the posterior** using :meth:`aspire.Aspire.sample_posterior`
   with either importance sampling or an adaptive SMC kernel (MiniPCN,
   BlackJAX, or custom samplers).
5. **Inspect, save, and reuse** the resulting
   :class:`aspire.samples.Samples`, :class:`aspire.history.History`
   objects, and the fitted flow.

Working with samples
--------------------

``aspire`` uses dataclasses defined in :mod:`aspire.samples` to keep sample
arrays, weights, and evidence estimates together. Key features:

* Automatic conversion between array namespaces (NumPy, JAX, PyTorch) via the
  ``xp`` argument.
* Convenience exporters (:meth:`aspire.samples.BaseSamples.to_dict`,
  :meth:`aspire.samples.BaseSamples.save`) for logging or serialisation.
* Plotting helpers (:meth:`aspire.samples.BaseSamples.plot_corner`) that
  integrate with ``corner`` while respecting weights.

When constructing your own samples, provide parameter names to enable labelled
plots and dataframes. Use :meth:`aspire.samples.Samples.from_samples` to
switch namespaces or merge multiple runs with
:meth:`aspire.samples.Samples.concatenate`.

Flows and transforms
--------------------

Aspire can work with any proposal that implements ``sample_and_log_prob`` and
``log_prob``; normalising flows remain the default. Flows are defined via
:class:`aspire.flows.base.Flow` and instantiated by
:meth:`aspire.Aspire.init_flow`. By default Aspire uses the ``zuko``
implementation of Masked Autoregressive Flows on top of PyTorch. The flow is
automatically wrapped with :class:`aspire.transforms.FlowTransform` (or a
composite of bounded / periodic transforms) so you can work with native
parameter ranges while still optimising in unconstrained space.

You can choose a backend by setting ``flow_backend="flowjax"`` to leverage JAX
or by providing a fully constructed ``flow`` instance. When ``flow_matching``
is enabled, Aspire trains a score-based model instead of a classical density
estimator (requires the `zuko` backend).

External flow implementations can be plugged in via the
``aspire.flows`` entry point group. See :ref:`custom_flows` for details.

Sampling strategies
-------------------

The :meth:`aspire.Aspire.sample_posterior` method orchestrates several
samplers, grouped below by inference style.

Importance sampling
~~~~~~~~~~~~~~~~~~~

``importance``
    Draws independent samples from the fitted flow and reweights them using
    the provided likelihood/prior functions. Perfect for quick sanity checks
    or sanity bounds on evidence estimates.

Markov chain Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~~

``minipcn``
    Runs the :class:`aspire.samplers.mcmc.MiniPCN` kernel directly (no SMC
    temperature ladder). Configure ``n_samples`` and pass MCMC kwargs such as
    ``n_steps`` or ``step_fn`` via ``sampler_kwargs``.
``emcee``
    Uses the :class:`aspire.samplers.mcmc.Emcee` ensemble sampler for
    gradient-free proposals. Provide ``sampler_kwargs`` like ``nwalkers`` or
    ``n_steps`` to control the chain length.

Sequential Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~

``smc`` / ``minipcn_smc``
    Runs adaptive SMC with the MiniPCN MCMC kernel. Configure the number of
    particles via ``n_samples`` and pass kernel settings in ``sampler_kwargs``
    (for example ``n_steps``, ``target_acceptance_rate`` or ``step_fn``).
``blackjax_smc``
    Uses BlackJAX kernels (requires the ``blackjax`` extra) while keeping the
    same adaptive temperature schedule as the MiniPCN backend.
``emcee_smc``
    Replaces the internal MCMC move with the ``emcee`` ensemble sampler,
    providing a gradient-free option that still benefits from SMC tempering.

You can plug in custom preconditioning by setting ``preconditioning`` to
``"standard"`` (affine normalisation based on current samples), ``"flow"``
(use the fitted flow as a transport map), or ``None`` to disable additional
transforms.

History, diagnostics, and persistence
-------------------------------------

Every sampler attaches a history object (see :mod:`aspire.history`) with
diagnostic metrics such as effective sample size, intermediate temperatures,
or acceptance rates. Plot them via :meth:`aspire.history.SMCHistory.plot` or
specialised helpers like :meth:`aspire.history.SMCHistory.plot_beta`.

Use the following methods to persist and later resume work:

* :meth:`aspire.Aspire.save_flow` / :meth:`aspire.Aspire.load_flow` to
  snapshot the trained flow.
* :meth:`aspire.Aspire.save_config` or
  :meth:`aspire.Aspire.save_config_to_json` to capture all hyperparameters.
* :meth:`aspire.samples.BaseSamples.save` to store weighted samples in HDF5
  for downstream analysis.

Together these utilities support iterative workflows where you continuously
refine the proposal distribution, reuse expensive likelihood evaluations, and
relaunch SMC runs with minimal boilerplate.
