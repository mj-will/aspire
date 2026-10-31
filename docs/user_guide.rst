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
   with importance sampling, MCMC, or adaptive SMC. Sequince owns the MCMC
   and SMC execution.
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

Aspire uses the flow implementations from ``sequince-density``, which implement
the normalizing-flow protocol defined by Sequince. By default Aspire uses its
``zuko`` implementation of Masked Autoregressive Flows on top of PyTorch. A
composition of :mod:`sequince.transforms` is attached to the flow so you can
work with native parameter ranges while still optimising in unconstrained
space.

You can choose a backend by setting ``flow_backend="flowjax"`` to leverage JAX
or by providing a fully constructed ``flow`` instance.

External flow implementations can be plugged in via the
``aspire.flows`` entry point group. See :ref:`custom_flows` for details.

Transform mechanics
~~~~~~~~~~~~~~~~~~~

Aspire keeps a clear separation between your native parameters and the space
where flows or kernels operate:

* A :class:`sequince.transforms.CompositeTransform` is attached to every flow
  created by :meth:`aspire.Aspire.init_flow`. By default, it maps bounded
  parameters to the real line with a logit and recentres / rescales dimensions with an affine
  transform learned from the training samples. Log-Jacobian terms are tracked so
  calls to ``log_prob`` or ``sample_and_log_prob`` remain properly normalised.
  ``bounded_to_unbounded`` and ``affine_transform`` can be specified when creating
  the Aspire instance to control this behaviour.
* The individual components are exposed by :mod:`sequince.transforms` for
  custom transports.

Preconditioning inside samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SMC and MCMC samplers also work in a transformed space. They fit the chosen
``preconditioning`` transform to the initial particles, perform moves there, and
then call ``inverse(...)`` (including the log-Jacobian) whenever the likelihood
or prior is evaluated. Configure it via
:meth:`aspire.Aspire.sample_posterior`:

* ``"default"`` / ``"standard"`` uses Sequince's
  :class:`sequince.transforms.CompositeTransform`
  with bounded-to-unbounded and affine scaling turned off by default; periodic
  wrapping still applies. To whiten dimensions or map bounds to the real line,
  pass ``preconditioning_kwargs={"affine_transform": True, "bounded_to_unbounded": True}``.
* ``"flow"`` fits a second density flow to the current particles and wraps it
  in :class:`sequince.transforms.FlowTransform` for SMC/MCMC updates. This
  reuses bounded handling while providing a richer geometry for the kernels.
  Periodic parameters receive the same bounded and affine treatment as other
  parameters in the flow. Their periodic topology is not modeled by the flow,
  and flow preconditioning does not apply periodic wrapping.
* ``"none"`` leaves the sampler in the original parameterisation with an identity
  transform. With ``preconditioning=None``, the importance sampler uses this
  identity transform; MCMC and SMC choose ``"default"`` instead.


.. note::

    By default, the preconditioning transform does not include bounded-to-unbounded
    steps. This means your log-prior and log-likelihood must handle points that
    lie outside the specified bounds (e.g. by returning ``-inf``). If you want
    the sampler to automatically map bounded parameters to an unconstrained
    space, enable the ``bounded_to_unbounded`` option in
    ``preconditioning_kwargs``.

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
    Runs a Sequince MiniPCN mutation pipeline directly (no SMC temperature
    ladder). Configure ``n_samples`` and pass MCMC options such as ``n_steps``
    or ``step_fn``.
``emcee``
    Runs Sequince's emcee mutation pipeline for gradient-free proposals.
    Options such as ``nwalkers`` and ``n_steps`` control the chain. Set
    ``vectorize=True`` to evaluate walkers in batches. This is particularly
    useful with JAX likelihoods and flows.

Sequential Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~

``smc`` / ``minipcn_smc``
    Runs adaptive SMC with the MiniPCN MCMC kernel. Configure the number of
    particles via ``n_samples`` and pass kernel settings in ``sampler_kwargs``
    (for example ``n_steps``, ``target_acceptance_rate`` or ``step_fn``).
``emcee_smc``
    Replaces the internal MCMC move with the ``emcee`` ensemble sampler,
    providing a gradient-free option that still benefits from SMC tempering.
    Use ``sampler_kwargs={"vectorize": True}`` for batched evaluation.
``blackjax_smc``
    Uses Sequince's BlackJAX mutation step for JAX-backed particles. Use
    ``flow_backend="flowjax"`` with JAX-compatible likelihood and prior callbacks.
    Install the ``blackjax`` extra and select ``nuts``, ``hmc``, or
    ``random-walk`` through ``sampler_kwargs``.

SMC limits and final samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``max_n_steps`` limits adaptive SMC iterations; it does not force the schedule to
reach beta one by increasing the minimum beta step. If a run stops early,
Aspire logs a warning and returns the intermediate target with its actual
``samples.beta``. Resume its checkpoint to continue towards the posterior.

The returned SMC population is resampled with replacement, including when the
requested final count equals the particle count. Reducing ``n_final_samples``
only resamples; increasing it resamples and mutates through Sequince's
``draw_additional_samples`` helper. ``sampler_kwargs["n_final_steps"]`` controls
that additional mutation and has no effect when reducing the count. These
choices change RNG consumption and can introduce duplicate output particles.

The previous ``beta_tolerance`` option is not accepted; use ``min_beta_step``
and ``max_beta_step`` to constrain increments. These bound the schedule and do
not provide a separate root-search tolerance.

History, diagnostics, and persistence
-------------------------------------

SMC samplers expose an Aspire :class:`aspire.history.SMCHistory` through
``aspire.sampler.history`` for compatibility with existing analysis code. It
contains temperatures, pre-resampling effective sample sizes, evidence
increments, mutation diagnostics, and optionally the particle population from
each iteration. The native Sequince result remains available through
``aspire.sampler.result``.

Standalone MCMC samplers continue to expose their native Sequince result
through both ``aspire.sampler.history`` and ``aspire.sampler.result``.

Use the following methods to persist and later resume work:

* :meth:`aspire.Aspire.save_flow` / :meth:`aspire.Aspire.load_flow` to
  snapshot the trained flow.
* :meth:`aspire.Aspire.save_config` or
  :meth:`aspire.Aspire.save_config_to_json` to capture all hyperparameters.
* :meth:`aspire.samples.BaseSamples.save` to store weighted samples in HDF5
  for downstream analysis.
* :meth:`aspire.Aspire.auto_checkpoint` to store Aspire configuration, its
  fitted proposal, and a native Sequince runtime checkpoint in one HDF5 file.

Samplers also accept native Sequince iteration callbacks through ``callbacks``.
Each callback receives ``(state, context, result)`` after an iteration. A
``checkpoint_callback`` instead receives a committed
``sequince.checkpoint.SMCCheckpoint`` at the ``checkpoint_every`` cadence.

Together these utilities support iterative workflows where you continuously
refine the proposal distribution, reuse expensive likelihood evaluations, and
relaunch SMC runs with minimal boilerplate.
