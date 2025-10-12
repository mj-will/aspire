import copy
import logging
from typing import Callable

import array_api_compat.torch as torch_api
import torch
import tqdm
import zuko
from array_api_compat import is_numpy_namespace, is_torch_array

from ...history import FlowHistory
from ...transforms import IdentityTransform
from ...utils import decode_dtype, encode_dtype, resolve_dtype
from ..base import Flow

logger = logging.getLogger(__name__)


class BaseTorchFlow(Flow):
    _flow = None
    xp = torch_api

    def __init__(
        self,
        dims: int,
        seed: int = 1234,
        device: str = "cpu",
        data_transform=None,
        dtype=None,
    ):
        resolved_dtype = (
            resolve_dtype(dtype, torch)
            if dtype is not None
            else torch.get_default_dtype()
        )
        if data_transform is None:
            data_transform = IdentityTransform(self.xp, dtype=resolved_dtype)
        elif getattr(data_transform, "dtype", None) is None:
            data_transform.dtype = resolved_dtype
        super().__init__(
            dims,
            device=torch.device(device or "cpu"),
            data_transform=data_transform,
        )
        self.dtype = resolved_dtype
        torch.manual_seed(seed)
        self.loc = None
        self.scale = None

    @property
    def flow(self):
        return self._flow

    @flow.setter
    def flow(self, flow):
        self._flow = flow
        self._flow.to(device=self.device, dtype=self.dtype)
        self._flow.compile()

    def fit(self, x) -> FlowHistory:
        raise NotImplementedError()

    def save(self, h5_file, path="flow"):
        """Save the weights of the flow to an HDF5 file."""
        from ...utils import recursively_save_to_h5_file

        flow_grp = h5_file.create_group(path)
        # Save config
        config = self.config_dict().copy()
        data_transform = config.pop("data_transform", None)
        dtype_value = config.get("dtype")
        if dtype_value is None:
            dtype_value = self.dtype
        else:
            dtype_value = resolve_dtype(dtype_value, torch)
        config["dtype"] = encode_dtype(torch, dtype_value)
        if data_transform is not None:
            data_transform.save(flow_grp, "data_transform")
        recursively_save_to_h5_file(flow_grp, "config", config)
        # Save weights
        weights_grp = flow_grp.create_group("weights")
        for name, tensor in self._flow.state_dict().items():
            weights_grp.create_dataset(name, data=tensor.cpu().numpy())

    @classmethod
    def load(self, h5_file, path="flow"):
        """Load the weights of the flow from an HDF5 file."""
        from ...utils import load_from_h5_file

        flow_grp = h5_file[path]
        # Load config
        config = load_from_h5_file(flow_grp, "config")
        config["dtype"] = decode_dtype(torch, config.get("dtype"))
        if "data_transform" in flow_grp:
            from ..transforms import BaseTransform

            data_transform = BaseTransform.load(
                flow_grp,
                "data_transform",
                strict=False,
            )
            config["data_transform"] = data_transform
        obj = self(**config)
        # Load weights
        weights = {
            name: torch.tensor(data[()])
            for name, data in flow_grp["weights"].items()
        }
        obj._flow.load_state_dict(weights)
        return obj


class ZukoFlow(BaseTorchFlow):
    def __init__(
        self,
        dims,
        flow_class: str | Callable = "MAF",
        data_transform=None,
        seed=1234,
        device: str = "cpu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            dims,
            device=device,
            data_transform=data_transform,
            seed=seed,
            dtype=dtype,
        )

        if isinstance(flow_class, str):
            FlowClass = getattr(zuko.flows, flow_class)
        else:
            FlowClass = flow_class

        # Ints are some times passed as strings, so we convert them
        if hidden_features := kwargs.pop("hidden_features", None):
            kwargs["hidden_features"] = list(map(int, hidden_features))

        self.flow = FlowClass(self.dims, 0, **kwargs)
        logger.info(f"Initialized normalizing flow: \n {self.flow}\n")

    def loss_fn(self, x):
        return -self.flow().log_prob(x).mean()

    def fit(
        self,
        x,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 500,
        validation_fraction: float = 0.2,
        clip_grad: float | None = None,
        lr_annealing: bool = False,
    ):
        from ...history import FlowHistory

        if not is_torch_array(x):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        else:
            x = torch.clone(x)
            x = x.type(self.dtype)
            x = x.to(self.device)
        x_prime = self.fit_data_transform(x)
        indices = torch.randperm(x_prime.shape[0])
        x_prime = x_prime[indices, ...]

        n = x_prime.shape[0]
        x_train = torch.as_tensor(
            x_prime[: -int(validation_fraction * n)],
            dtype=self.dtype,
            device=self.device,
        )

        logger.info(
            f"Training on {x_train.shape[0]} samples, "
            f"validating on {x_prime.shape[0] - x_train.shape[0]} samples."
        )

        if torch.isnan(x_train).any():
            dims_with_nan = (
                torch.isnan(x_train).any(dim=0).nonzero(as_tuple=True)[0]
            )
            raise ValueError(
                f"Training data contains NaN values in dimensions: {dims_with_nan.tolist()}"
            )
        if not torch.isfinite(x_train).all():
            dims_with_inf = (
                (~torch.isfinite(x_train)).any(dim=0).nonzero(as_tuple=True)[0]
            )
            raise ValueError(
                f"Training data contains infinite values in dimensions: {dims_with_inf.tolist()}"
            )

        x_val = torch.as_tensor(
            x_prime[-int(validation_fraction * n) :],
            dtype=self.dtype,
            device=self.device,
        )
        if torch.isnan(x_val).any():
            raise ValueError("Validation data contains infinite values.")

        if not torch.isfinite(x_val).all():
            raise ValueError("Validation data contains infinite values.")

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
        if lr_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, n_epochs
            )
        history = FlowHistory()

        best_val_loss = float("inf")
        best_flow_state = None

        with tqdm.tqdm(range(n_epochs), desc="Epochs") as pbar:
            for _ in pbar:
                self.flow.train()
                loss_epoch = 0.0
                for (x_batch,) in dataset:
                    loss = self.loss_fn(x_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.flow.parameters(), clip_grad
                        )
                    optimizer.step()
                    loss_epoch += loss.item()
                if lr_annealing:
                    scheduler.step()
                avg_train_loss = loss_epoch / len(dataset)
                history.training_loss.append(avg_train_loss)
                self.flow.eval()
                val_loss = 0.0
                for (x_batch,) in val_dataset:
                    with torch.no_grad():
                        val_loss += self.loss_fn(x_batch).item()
                avg_val_loss = val_loss / len(val_dataset)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_flow_state = copy.deepcopy(self.flow.state_dict())

                history.validation_loss.append(avg_val_loss)
                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{avg_val_loss:.4f}",
                )
        if best_flow_state is not None:
            self.flow.load_state_dict(best_flow_state)
            logger.info(f"Loaded best model with val loss {best_val_loss:.4f}")

        self.flow.eval()
        return history

    def sample_and_log_prob(self, n_samples: int, xp=torch_api):
        with torch.no_grad():
            x_prime, log_prob = self.flow().rsample_and_log_prob((n_samples,))
        x, log_abs_det_jacobian = self.inverse_rescale(x_prime)
        return xp.asarray(x), xp.asarray(log_prob - log_abs_det_jacobian)

    def sample(self, n_samples: int, xp=torch_api):
        with torch.no_grad():
            x_prime = self.flow().rsample((n_samples,))
        x = self.inverse_rescale(x_prime)[0]
        return xp.asarray(x)

    def log_prob(self, x, xp=torch_api):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        x_prime, log_abs_det_jacobian = self.rescale(x)
        return xp.asarray(
            self._flow().log_prob(x_prime) + log_abs_det_jacobian
        )

    def forward(self, x, xp=torch_api):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        x_prime, log_j_rescale = self.rescale(x)
        z, log_abs_det_jacobian = self._flow().transform.call_and_ladj(x_prime)
        if is_numpy_namespace(xp):
            # Convert to numpy namespace if needed
            z = z.detach().numpy()
            log_abs_det_jacobian = log_abs_det_jacobian.detach().numpy()
            log_j_rescale = log_j_rescale.detach().numpy()
        return xp.asarray(z), xp.asarray(log_abs_det_jacobian + log_j_rescale)

    def inverse(self, z, xp=torch_api):
        z = torch.as_tensor(z, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            x_prime, log_abs_det_jacobian = (
                self._flow().transform.inv.call_and_ladj(z)
            )
        x, log_j_rescale = self.inverse_rescale(x_prime)
        if is_numpy_namespace(xp):
            # Convert to numpy namespace if needed
            x = x.detach().numpy()
            log_abs_det_jacobian = log_abs_det_jacobian.detach().numpy()
            log_j_rescale = log_j_rescale.detach().numpy()
        return xp.asarray(x), xp.asarray(log_j_rescale + log_abs_det_jacobian)


class ZukoFlowMatching(ZukoFlow):
    def __init__(
        self,
        dims,
        data_transform=None,
        seed=1234,
        device="cpu",
        eta: float = 1e-3,
        dtype=None,
        **kwargs,
    ):
        kwargs.setdefault("hidden_features", 4 * [100])
        super().__init__(
            dims,
            seed=seed,
            device=device,
            data_transform=data_transform,
            flow_class="CNF",
            dtype=dtype,
        )
        self.eta = eta

    def loss_fn(self, theta: torch.Tensor):
        t = torch.rand(
            theta.shape[:-1], dtype=theta.dtype, device=theta.device
        )
        t_ = t[..., None]
        eps = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * eps
        v = eps - theta
        return (self._flow.transform.f(t, theta_prime) - v).square().mean()
