import pytest

from aspire.aspire import Aspire
from aspire.samplers.mcmc import Emcee, MiniPCN
from aspire.samplers.smc import BlackJAXSMC, EmceeSMC, MiniPCNSMC


def test_sampler_names_resolve_to_expected_classes(sampler_kwargs):
    aspire = Aspire(
        log_likelihood=sampler_kwargs["log_likelihood"],
        log_prior=sampler_kwargs["log_prior"],
        dims=2,
        xp=sampler_kwargs["xp"],
    )

    assert aspire.get_sampler_class("smc") is MiniPCNSMC
    assert aspire.get_sampler_class("minipcn_smc") is MiniPCNSMC
    assert aspire.get_sampler_class("emcee_smc") is EmceeSMC
    assert aspire.get_sampler_class("blackjax_smc") is BlackJAXSMC
    assert aspire.get_sampler_class("minipcn") is MiniPCN
    assert aspire.get_sampler_class("emcee") is Emcee


@pytest.mark.parametrize("backend", ["zuko", "flowjax"])
@pytest.mark.parametrize("bounded_to_unbounded", [False, True])
def test_periodic_flow_preconditioning_treats_coordinates_as_nonperiodic(
    sampler_kwargs, backend: str, bounded_to_unbounded: bool
) -> None:
    pytest.importorskip(f"sequince_density.flows.{backend}")
    xp = sampler_kwargs["xp"]
    aspire = Aspire(
        log_likelihood=sampler_kwargs["log_likelihood"],
        log_prior=sampler_kwargs["log_prior"],
        dims=2,
        parameters=["angle", "x"],
        periodic_parameters=["angle"],
        prior_bounds={"angle": (0.0, 2.0), "x": (0.0, 2.0)},
        flow=sampler_kwargs["prior_flow"],
        flow_backend=backend,
        xp=xp,
    )
    sampler = aspire.init_sampler(
        "smc",
        preconditioning="flow",
        preconditioning_kwargs={
            "bounded_to_unbounded": bounded_to_unbounded,
        },
    )
    transform = sampler.preconditioning_transform.flow.particle_transform
    particles = xp.asarray([[0.5, 0.5], [1.5, 1.5]])
    values, log_det, auxiliary = transform.forward(particles)
    assert bool(xp.all(xp.abs(values[:, 0] - values[:, 1]) < 1e-7))
    if bounded_to_unbounded:
        expected = xp.log(particles / (2.0 - particles))
    else:
        expected = particles
    assert bool(xp.all(xp.abs(values - expected) < 1e-7))
    restored, inverse_log_det = transform.inverse(values, auxiliary=auxiliary)
    assert bool(xp.all(xp.abs(restored - particles) < 1e-7))
    assert bool(xp.all(xp.abs(log_det + inverse_log_det) < 1e-7))

    # Standard preconditioning still wraps the declared periodic coordinate.
    standard = aspire.init_sampler("smc", preconditioning="default")
    outside = xp.asarray([[2.5, 2.5]])
    if not bounded_to_unbounded:
        unchanged, _, _ = transform.forward(outside)
        assert bool(xp.all(xp.abs(unchanged - outside) < 1e-7))
    wrapped, _, _ = standard.preconditioning_transform.forward(outside)
    assert bool(xp.all(xp.abs(wrapped - xp.asarray([[0.5, 2.5]])) < 1e-7))
