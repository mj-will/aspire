Installation
============

``aspire`` targets Python 3.10+ and relies on ``numpy``, ``matplotlib``,
``array-api-compat`` and ``h5py`` for core functionality. Optional extras
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
and optionally the ``minipcn`` SMC kernel.

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
     - PyTorch + ``zuko`` backend (default) for normalizing flows and flow matching.
   * - ``minipcn``
     - Enables the MiniPCN SMC kernel.
   * - ``emcee``
     - Enables the ``emcee`` ensemble sampler integration.
   * - ``blackjax``
     - Enables the BlackJAX SMC kernel.
   * - ``test``
     - Installs ``pytest`` and coverage helpers for local testing.

Install extras via:

.. code-block:: console

   $ python -m pip install "aspire-inference[torch,minipcn]"

From source
-----------

Clone the repository and install in editable mode:

.. code-block:: console

   $ git clone https://github.com/mj-will/aspire.git
   $ cd aspire
    # (optional) create/activate a virtual environment
   $ python -m pip install -e ".[torch,minipcn]"

After installation, run the unit test suite to confirm everything is wired up:

.. code-block:: console

   $ python -m pytest

Building the docs locally requires ``sphinx`` and (optionally) the
``sphinx-``. These are installed automatically when you run
``python -m pip install -r docs/requirements.txt`` if such a file exists, or
you can install ``sphinx`` manually before invoking ``make html`` inside the
``docs`` directory.
