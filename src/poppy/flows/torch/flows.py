import torch
import tqdm
import zuko

from ..base import Flow


class BaseTorchFlow(Flow):
    def __init__(self, dims: int, seed: int = 1234, device: str = "cpu"):
        super().__init__(dims)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.loc = None
        self.scale = None

    def fit(self, x):
        self.fit_rescaling(x)

    def fit_rescaling(self, x: torch.Tensor) -> None:
        self.loc = torch.mean(x, axis=0)
        self.scale = torch.std(x, axis=0)
        self.log_abs_det_jacobian = torch.sum(torch.log(self.scale))

    def rescale(self, x):
        return (
            x - self.loc
        ) / self.scale, self.log_abs_det_jacobian * torch.ones(
            x.shape[0], device=x.device
        )

    def inverse_rescale(self, x_prime):
        return (
            x_prime * self.scale + self.loc,
            -self.log_abs_det_jacobian
            * torch.ones(x_prime.shape[0], device=x_prime.device),
        )


class ZukoFlow(BaseTorchFlow):
    def __init__(self, dims, seed=1234, device: str = "cpu", **kwargs):
        super().__init__(dims, seed, device)
        self._flow = zuko.flows.MAF(self.dims, 0, **kwargs)
        self._flow.compile()

    def loss_fn(self, x):
        return -self._flow().log_prob(x).mean()

    def fit(
        self,
        x,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 500,
        validation_fraction: float = 0.2,
    ):
        from ...history import History

        super().fit(x)

        x_prime, _ = self.rescale(x)
        n = x_prime.shape[0]
        x_train_numpy = x_prime[: -int(validation_fraction * n)]
        x_val_numpy = x_prime[-int(validation_fraction * n) :]

        x_train = torch.FloatTensor(x_train_numpy)
        x_val = torch.FloatTensor(x_val_numpy)

        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train),
            shuffle=True,
            batch_size=batch_size,
        )
        val_dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_val),
            shuffle=False,
            batch_size=batch_size,
        )

        # Train to maximize the log-likelihood
        optimizer = torch.optim.Adam(self._flow.parameters(), lr=lr)
        history = History()

        for _ in tqdm.tqdm(range(n_epochs)):
            self._flow.train()
            loss_epoch = 0.0
            for (x_batch,) in dataset:
                loss = self.loss_fn(x_batch)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
                optimizer.step()
                loss_epoch += loss.item()
            # scheduler.step()
            history.training_loss.append(loss_epoch / len(dataset))
            self._flow.eval()
            val_loss = 0.0
            for (x_batch,) in val_dataset:
                with torch.inference_mode():
                    val_loss += self.loss_fn(x_batch).item()
            history.validation_loss.append(val_loss / len(val_dataset))
        return history

    def sample(self, n_samples: int):
        with torch.no_grad():
            x_prime, log_prob = self._flow().rsample_and_log_prob((n_samples,))
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return x, log_prob + log_abs_det_jacobian

    def log_prob(self, x):
        x_prime, log_abs_det_jacobian = self.rescale(x)
        return self._flow().log_prob(x_prime) + log_abs_det_jacobian


class ZukoFlowMatching(ZukoFlow):
    def __init__(
        self, dims, seed=1234, device="cpu", eta: float = 1e-3, **kwargs
    ):
        super().__init__(dims, seed, device)
        self.eta = eta
        kwargs.setdefault("hidden_features", 4 * [100])
        self._flow = zuko.flows.CNF(self.dims, 0, **kwargs)

    def loss_fn(self, theta: torch.Tensor):
        t = torch.rand(
            theta.shape[:-1], dtype=theta.dtype, device=theta.device
        )
        t_ = t[..., None]
        eps = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * eps
        v = eps - theta
        return (self._flow.transform.f(t, theta_prime) - v).square().mean()
