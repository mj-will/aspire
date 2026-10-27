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


def test_periodic_flow_preconditioning_is_rejected(sampler_kwargs):
    aspire = Aspire(
        log_likelihood=sampler_kwargs["log_likelihood"],
        log_prior=sampler_kwargs["log_prior"],
        dims=2,
        parameters=["angle", "x"],
        periodic_parameters=["angle"],
        prior_bounds={"angle": (0, 6.28)},
        xp=sampler_kwargs["xp"],
    )
    with pytest.raises(NotImplementedError, match="periodic parameters"):
        aspire.init_sampler("smc", preconditioning="flow")
