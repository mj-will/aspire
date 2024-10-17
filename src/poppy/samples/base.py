from dataclasses import dataclass, field


@dataclass
class BaseSamples:
    x: object
    parameters: list[str] | None = None
    log_likelihood: object | None = None
    log_prior: object | None = None
    log_q: object | None = None
    log_w: object = field(init=False)
    weights: object = field(init=False)
    evidence: float = field(init=False)
    log_evidence: float = field(init=False)

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = [f"x_{i}" for i in range(len(self.x[0]))]
        if all(
            x is not None
            for x in [self.log_likelihood, self.log_prior, self.log_q]
        ):
            self.compute_weights()
        else:
            self.log_w = None
            self.weights = None
            self.evidence = None
            self.log_evidence = None
            self.evidence_error = None
            self.log_evidence_error = None

    @property
    def efficiency(self):
        return self.effective_sample_size / len(self.x)

    def rejection_sample(self, **kwargs):
        raise NotImplementedError

    def compute_weights(self):
        raise NotImplementedError

    def to_numpy(self):
        raise NotImplementedError

    def to_jax(self):
        raise NotImplementedError

    def to_torch(self):
        raise NotImplementedError

    def to_dict(self, flat: bool = True):
        samples = dict(zip(self.parameters, self.x.T, strict=True))
        out = {
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "log_q": self.log_q,
            "log_w": self.log_w,
            "weights": self.weights,
            "evidence": self.evidence,
            "log_evidence": self.log_evidence,
            "evidence_error": self.evidence_error,
            "log_evidence_error": self.log_evidence_error,
            "effective_sample_size": self.effective_sample_size,
        }
        if flat:
            out.update(samples)
        else:
            out["samples"] = samples
        return out

    def to_dataframe(self, flat: bool = True):
        import pandas as pd

        return pd.DataFrame(self.to_dict(flat=flat))

    def to_backend(self):
        from ..backend import get_backend

        backend = get_backend().name
        if backend == "numpy":
            return self.to_numpy()
        elif backend == "jax":
            return self.to_jax()
        elif backend == "torch":
            return self.to_torch()
        else:
            raise ValueError(
                f"Unknown backend {backend}. Known backends are numpy, jax, and torch."
            )
