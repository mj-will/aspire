from ..samples import Samples
from ..utils import track_calls
from .base import Sampler


class ImportanceSampler(Sampler):
    @track_calls
    def sample(self, n_samples: int) -> Samples:
        x, log_q = self.prior_flow.sample_and_log_prob(n_samples)
        samples = Samples(
            x, log_q=log_q, xp=self.xp, parameters=self.parameters
        )
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        samples.compute_weights()
        return samples
