class Flow:
    def __init__(self, dims: int):
        self.dims = dims

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, n_samples):
        raise NotImplementedError

    def fit(self, samples, **kwargs):
        raise NotImplementedError

    def fit_rescaling(self, samples):
        raise NotImplementedError

    def rescale(self, samples):
        raise NotImplementedError

    def inverse_rescale(self, samples):
        raise NotImplementedError
