Entry Points
============

Aspire uses Python's entry point system to allow for extensibility and
integration with external libraries. This mechanism enables users to register
custom components that can be seamlessly integrated into the Aspire framework.

.. _custom_flows:

Custom Flows
------------

Aspire supports custom flow implementations via the
``aspire.flows`` entry point group. To register a new flow backend, define an
entry point in your ``pyproject.toml`` like so:

.. code-block:: toml

    [project.entry-points."aspire.flows"]
    myflow = "my_module:MyFlowClass"


The specified class must implement :class:`sequince.flows.NormalizingFlow` (or
inherit from ``sequince_density.flows.base.Flow``), implement the required
methods, and define the ``xp`` attribute which specifies the array namespace.
You can then select your custom flow by setting
``flow_backend="myflow"`` when initializing Aspire.

Provide ``sample_and_log_prob`` for direct flow sampling and ``state_dict`` /
``load_state_dict`` for persistence. Subclassing the Sequince Density base class
also records constructor arguments for Aspire's flow loader. The entry-point
group name itself is unchanged.

Existing external flow instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrap an old-style Aspire flow before passing it to Aspire::

    from aspire import Aspire
    from aspire.compat import LegacyFlowAdapter

    flow = LegacyFlowAdapter(my_existing_flow)
    analysis = Aspire(..., flow=flow)

The adapter bridges methods without ``context`` or ``rng`` arguments and
retains the ``FlowHistory`` returned by legacy training. Density and transform
methods must already include their data transforms and Jacobians. Supplied
instances with legacy signatures are also adapted automatically, with a
deprecation warning encouraging explicit wrapping. This applies to sampler
``proposal`` arguments too.

Methods without an RNG argument retain their own RNG behavior; supplying a
Sequince RNG does not make those methods reproducible. Flow preconditioning
requires the legacy flow to implement the complete ``forward`` and ``inverse``
maps.

Persistence uses the legacy ``state_dict`` / ``load_state_dict`` pair or its
HDF5 ``save`` / ``load`` methods. Constructor arguments are inferred from
``_init_args`` or ``config_dict()``. If neither is available, supply them as
``LegacyFlowAdapter(my_existing_flow, flow_kwargs={...})``. The adapter supports
existing instances; it does not restore removed ``aspire.flows`` imports or
make old checkpoint files compatible.
