Installation
============

``aspire`` targets Python 3.11+ and relies on ``sequince``, ``sequince-density``, ``numpy``,
``matplotlib``, ``array-api-compat`` and ``h5py``. Optional extras
provide tighter integration with popular samplers and flow backends.

Basic setup
-----------

Install the library from PyPI (note the published name):

.. code-block:: console

   $ python -m pip install aspire-inference

The installed distribution exposes the ``aspire`` import namespace. By default,
this doesn't include any optional dependencies beyond the core ones listed above.
We recommend installing with at least one backend for normalizing flows, e.g.
``torch`` (PyTorch + ``zuko``) or ``jax`` (JAX + ``flowjax``).
Install the optional ``minipcn`` extra for Aspire's default ``smc`` sampler
or the ``minipcn`` and ``minipcn_smc`` samplers.

Optional extras
---------------

Additional features can be enabled by installing the relevant extras:

.. list-table::
   :header-rows: 1
   :widths: 20 70

   * - Extra
     - Purpose
   * - ``scipy``
     - Access to SciPy utilities used by certain transforms.
   * - ``jax``
     - JAX + ``flowjax`` backend for training normalizing flows.
   * - ``torch``
     - PyTorch + ``zuko`` backend (default) for normalizing flows.
   * - ``minipcn``
     - MiniPCN mutation for the ``smc``, ``minipcn_smc`` and ``minipcn`` samplers.
   * - ``emcee``
     - Enables the ``emcee`` ensemble sampler integration.
   * - ``blackjax``
     - Enables BlackJAX mutation for JAX-backed SMC runs.
   * - ``test``
     - Installs ``pytest`` and coverage helpers for local testing.

Install extras via:

.. code-block:: console

   $ python -m pip install "aspire-inference[torch,minipcn]"

Development installation
------------------------

For an editable development installation, Pixi setup, tests, and documentation
build instructions, see the :doc:`contribution guide <contributing>`.
