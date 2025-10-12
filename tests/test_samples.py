# test_samples.py
import math
import pickle

import numpy as np
import pytest

from aspire.samples import BaseSamples, Samples, SMCSamples


def make_simple_samples(n=5, d=2, a=1.234):
    """
    Construct a simple, consistent set of arrays:
    - x: shape (n, d)
    - log_likelihood = a (constant)
    - log_prior = 0
    - log_q = 0
    This makes the math easy: evidence = exp(a), ESS = n, etc.
    """
    x = np.arange(n * d, dtype=float).reshape(n, d)
    ll = np.full(n, a, dtype=float)
    lp = np.zeros(n, dtype=float)
    lq = np.zeros(n, dtype=float)
    return x, ll, lp, lq


# -------- BaseSamples ---------------------------------------------------------


def test_basesamples_defaults_and_dims_numpy_namespace():
    x = np.zeros((3, 4))
    bs = BaseSamples(x=x)
    # xp inferred from x
    assert bs.xp is not None
    assert bs.dtype == x.dtype
    assert bs.x.dtype == x.dtype
    # parameters default names
    assert bs.parameters == ["x_0", "x_1", "x_2", "x_3"]
    # dims computed correctly
    assert bs.dims == 4
    # device set for numpy
    assert bs.device == "cpu"
    # __len__
    assert len(bs) == 3


def test_basesamples_to_dict_flat_and_nested():
    x = np.arange(6.0).reshape(3, 2)
    ll = np.array([0.1, 0.2, 0.3])
    bs = BaseSamples(x=x, log_likelihood=ll, parameters=["a", "b"])

    assert bs.log_likelihood.dtype == bs.dtype

    d_flat = bs.to_dict(flat=True)
    assert "a" in d_flat and "b" in d_flat
    assert "log_likelihood" in d_flat and np.allclose(
        d_flat["log_likelihood"], ll
    )

    d_nested = bs.to_dict(flat=False)
    assert "samples" in d_nested and isinstance(d_nested["samples"], dict)
    assert set(d_nested["samples"].keys()) == {"a", "b"}


def test_basesamples_to_numpy_and_namespace_roundtrip():
    x = np.arange(6.0).reshape(3, 2)
    ll = np.array([0.1, 0.2, 0.3])
    bs = BaseSamples(x=x, log_likelihood=ll, parameters=["a", "b"])

    bs_np = bs.to_numpy()
    assert isinstance(bs_np.x, np.ndarray)
    assert isinstance(bs_np.log_likelihood, np.ndarray)

    # Convert to the same namespace (numpy) explicitly
    bs_ns = bs.to_namespace(np)
    assert isinstance(bs_ns.x, np.ndarray)
    assert isinstance(bs_ns.log_likelihood, np.ndarray)


def test_basesamples_getitem_slice_returns_same_type_and_fields():
    x = np.arange(12.0).reshape(6, 2)
    ll = np.linspace(0, 1, 6)
    lp = np.linspace(1, 2, 6)
    lq = np.linspace(-1, 0, 6)
    bs = BaseSamples(
        x=x, log_likelihood=ll, log_prior=lp, log_q=lq, parameters=["a", "b"]
    )

    one = bs[2]
    assert isinstance(one, BaseSamples.__mro__[0]) or isinstance(
        one, BaseSamples
    )  # basic sanity
    assert one.x.shape == (2,)  # row
    assert one.parameters == ["a", "b"]
    assert np.isclose(one.log_likelihood, ll[2])
    assert np.isclose(one.log_prior, lp[2])
    assert np.isclose(one.log_q, lq[2])
    assert one.dtype == bs.dtype


def test_basesamples_respects_explicit_dtype_string():
    x = np.arange(6).reshape(3, 2)
    ll = np.linspace(0.0, 1.0, 3)
    bs = BaseSamples(x=x, log_likelihood=ll, dtype="float32")
    assert bs.x.dtype == np.dtype("float32")
    assert bs.dtype == np.dtype("float32")
    assert bs.log_likelihood.dtype == np.dtype("float32")


def test_basesamples_pickle_restores_namespace_and_fields():
    x = np.arange(6.0).reshape(3, 2)
    bs = BaseSamples(x=x)
    blob = pickle.dumps(bs)
    bs2 = pickle.loads(blob)
    # xp restored from x
    assert bs2.xp is not None
    # functionality intact
    assert bs2.dims == 2
    assert np.allclose(bs2.x, x)


# -------- Samples -------------------------------------------------------------


def test_samples_compute_weights_constant_case():
    n, d = 10, 3
    a = 1.5
    x, ll, lp, lq = make_simple_samples(n=n, d=d, a=a)
    s = Samples(
        x=x,
        log_likelihood=ll,
        log_prior=lp,
        log_q=lq,
        parameters=[f"p{i}" for i in range(d)],
    )

    # log_w = a; evidence = exp(a); evidence_error = 0; ESS = n
    assert np.allclose(s.log_w, a)
    assert math.isclose(float(s.evidence), math.exp(a), rel_tol=1e-12)
    assert math.isclose(float(s.log_evidence), a, rel_tol=1e-12)
    assert math.isclose(float(s.evidence_error), 0.0, abs_tol=1e-15)
    assert math.isclose(float(s.log_evidence_error), 0.0, abs_tol=1e-15)
    assert math.isclose(float(s.effective_sample_size), n, rel_tol=1e-12)
    assert math.isclose(float(s.efficiency), 1.0, rel_tol=1e-12)

    # scaled_weights ∝ exp(log_w - max)
    sw = s.scaled_weights
    assert np.allclose(sw, np.exp(a - a))  # all ones


def test_samples_to_dict_includes_weight_fields():
    x, ll, lp, lq = make_simple_samples(n=4, d=2, a=0.3)
    s = Samples(
        x=x, log_likelihood=ll, log_prior=lp, log_q=lq, parameters=["a", "b"]
    )
    d = s.to_dict(flat=True)
    for k in [
        "log_w",
        "weights",
        "evidence",
        "log_evidence",
        "evidence_error",
        "log_evidence_error",
        "effective_sample_size",
        "a",
        "b",
    ]:
        assert k in d


def test_samples_getitem_preserves_evidence_fields():
    x, ll, lp, lq = make_simple_samples(n=6, d=2, a=0.2)
    s = Samples(
        x=x, log_likelihood=ll, log_prior=lp, log_q=lq, parameters=["a", "b"]
    )
    s2 = s[1:4]
    assert isinstance(s2, Samples)
    assert s2.x.shape[0] == 3
    # evidence fields were set on s; __getitem__ preserves them via from_samples
    assert math.isclose(
        float(s2.log_evidence), float(s.log_evidence), rel_tol=1e-12
    )
    assert math.isclose(
        float(s2.log_evidence_error),
        float(s.log_evidence_error),
        rel_tol=1e-12,
    )


def test_samples_concatenate_success_and_mismatch_errors():
    x1, ll1, lp1, lq1 = make_simple_samples(n=3, d=2, a=0.0)
    x2, ll2, lp2, lq2 = make_simple_samples(n=4, d=2, a=0.1)
    s1 = Samples(
        x=x1,
        log_likelihood=ll1,
        log_prior=lp1,
        log_q=lq1,
        parameters=["a", "b"],
    )
    s2 = Samples(
        x=x2,
        log_likelihood=ll2,
        log_prior=lp2,
        log_q=lq2,
        parameters=["a", "b"],
    )

    sc = Samples.concatenate([s1, s2])
    assert isinstance(sc, Samples)
    assert sc.x.shape == (7, 2)
    assert sc.parameters == ["a", "b"]

    # Mismatching dtype -> error
    x3 = x1.astype(np.float32)
    s3 = Samples(
        x=x3,
        log_likelihood=ll1,
        log_prior=lp1,
        log_q=lq1,
        parameters=["a", "b"],
        dtype="float32",
    )
    with pytest.raises(ValueError):
        Samples.concatenate([s1, s3])

    # Mismatching parameters -> error
    t_bad = Samples(
        x=x2,
        log_likelihood=ll2,
        log_prior=lp2,
        log_q=lq2,
        parameters=["a", "c"],
    )
    with pytest.raises(ValueError):
        Samples.concatenate([s1, t_bad])

    # Mismatching namespaces -> error (simulate by converting to a different namespace if available)
    # Here both are numpy, so simulate by monkeypatching xp check with a subclass of BaseSamples? Easiest:
    # Create a dummy with same params but fake xp attr.
    s_fake = Samples(
        x=x2,
        log_likelihood=ll2,
        log_prior=lp2,
        log_q=lq2,
        parameters=["a", "b"],
    )
    s_fake.xp = object()  # break equality
    with pytest.raises(ValueError):
        Samples.concatenate([s1, s_fake])


def test_samples_rejection_sample_accepts_proportionally():
    # Make one point much heavier
    n, d = 100, 1
    x = np.arange(n, dtype=float).reshape(n, d)
    ll = np.zeros(n)
    lp = np.zeros(n)
    lq = np.zeros(n)
    ll[0] = 10.0  # very large log-likelihood for index 0
    s = Samples(
        x=x, log_likelihood=ll, log_prior=lp, log_q=lq, parameters=["p0"]
    )
    rng = np.random.default_rng(1234)
    rs = s.rejection_sample(rng=rng)
    assert isinstance(rs, Samples)
    assert len(rs) <= len(s)
    # Heaviest point should be present with high probability
    if len(rs) > 0:
        assert 0 in rs.x.squeeze().astype(int)


def test_smc_unnormalized_and_normalized_log_weights_and_ratio():
    n, d = 6, 2
    x = np.arange(n * d, dtype=float).reshape(n, d)
    ll = np.linspace(0.0, 1.0, n)
    lp = np.zeros(n)
    lq = np.zeros(n)
    beta0 = 0.3
    smc = SMCSamples(
        x=x,
        log_likelihood=ll,
        log_prior=lp,
        log_q=lq,
        beta=beta0,
        parameters=["a", "b"],
    )

    beta1 = 0.7
    # unnormalized weights (beta-beta0) * (ll+lp) since lq=0
    uw = smc.unnormalized_log_weights(beta1)
    assert np.allclose(uw, (beta1 - beta0) * (ll + lp))

    # log_evidence_ratio = logsumexp(uw) - log(n)
    import scipy.special as sps  # optional; if unavailable, reimplement quickly

    lse = sps.logsumexp(uw) - math.log(n)
    assert math.isclose(
        float(smc.log_evidence_ratio(beta1)), float(lse), rel_tol=1e-12
    )

    # normalized (shifted) log weights should not contain NaN and sum of exp(log_w - logsumexp(log_w)) = 1
    lw = smc.log_weights(beta1)
    w = np.exp(lw - sps.logsumexp(lw))
    assert np.isfinite(w).all()
    assert math.isclose(w.sum(), 1.0, rel_tol=1e-12)


def test_smc_log_weights_nan_guard_raises():
    x = np.zeros((3, 1))
    ll = np.array([0.0, np.nan, 1.0])
    smc = SMCSamples(
        x=x,
        log_likelihood=ll,
        log_prior=np.zeros(3),
        log_q=np.zeros(3),
        beta=0.0,
    )
    with pytest.raises(ValueError):
        _ = smc.log_weights(0.5)


def test_smc_resample_shapes_and_beta_and_warning_on_same_beta(caplog):
    n, d = 10, 2
    x = np.arange(n * d, dtype=float).reshape(n, d)
    ll = np.linspace(0, 1, n)
    lp = np.zeros(n)
    lq = np.zeros(n)
    beta0 = 0.2
    smc = SMCSamples(
        x=x,
        log_likelihood=ll,
        log_prior=lp,
        log_q=lq,
        beta=beta0,
        parameters=["a", "b"],
    )

    rng = np.random.default_rng(42)
    out = smc.resample(beta=0.8, n_samples=7, rng=rng)
    assert isinstance(out, SMCSamples)
    assert out.x.shape == (7, d)
    assert math.isclose(out.beta, 0.8)

    # same beta & n_samples=None -> returns self and logs a warning
    caplog.clear()
    with caplog.at_level("WARNING"):
        same = smc.resample(beta=beta0, n_samples=None, rng=rng)
    assert same is smc
    assert any(
        "Resampling with the same beta value" in r.message
        for r in caplog.records
    )


def test_smc_variance_delta_method_nonnegative():
    x = np.arange(10.0).reshape(5, 2)
    ll = np.linspace(0, 1, 5)
    smc = SMCSamples(
        x=x,
        log_likelihood=ll,
        log_prior=np.zeros(5),
        log_q=np.zeros(5),
        beta=0.3,
    )
    var = smc.log_evidence_ratio_variance(beta=0.6)
    assert float(var) >= 0.0 or np.isnan(var)


def test_smc_to_standard_samples_and_getitem_preserve_fields():
    x = np.arange(8.0).reshape(4, 2)
    ll = np.linspace(0, 1, 4)
    smc = SMCSamples(
        x=x,
        log_likelihood=ll,
        log_prior=np.zeros(4),
        log_q=np.zeros(4),
        beta=0.4,
        parameters=["a", "b"],
        log_evidence=1.23,
    )

    std = smc.to_standard_samples()
    assert isinstance(std, Samples)
    # For std Samples, weights aren't computed (log_q missing by design),
    # but log_evidence is carried over.
    assert std.log_w is None
    assert math.isclose(float(std.log_evidence), 1.23, rel_tol=1e-12)

    # __getitem__ preserves beta and log_evidence
    smc2 = smc[1:3]
    assert isinstance(smc2, SMCSamples)
    assert smc2.x.shape[0] == 2
    assert math.isclose(float(smc2.beta), 0.4, rel_tol=1e-12)
    assert math.isclose(float(smc2.log_evidence), 1.23, rel_tol=1e-12)


def test_str_contains_counts_and_metrics():
    x, ll, lp, lq = make_simple_samples(n=4, d=2, a=0.7)
    s = Samples(
        x=x, log_likelihood=ll, log_prior=lp, log_q=lq, parameters=["a", "b"]
    )
    s_str = str(s)
    assert "No. samples: 4" in s_str
    assert "No. parameters: 2" in s_str
    assert "Log evidence:" in s_str
    assert "Effective sample size:" in s_str
    assert "Efficiency:" in s_str
