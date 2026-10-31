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
also records constructor arguments for Aspire's flow loader. Plugins written
against the removed Aspire ``Flow`` base class need to adopt these interfaces;
the entry-point group name itself is unchanged.
