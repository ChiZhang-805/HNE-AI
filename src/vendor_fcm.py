"""Original FCM architecture and minimax/context objectives on daily windows.

Preserves the existing disclosed compatibility repair: predicted context stays
detached, but its learned context projection is allowed to receive gradients.
"""
from __future__ import annotations
from contextlib import contextmanager
import importlib
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

VENDOR = Path(__file__).resolve().parents[1] / "third_party/FCM"
_MODEL = None


@contextmanager
def author_imports():
    owned = lambda name: name.split(".", 1)[0] in {"models", "layers", "utils"}
    previous = {k: v for k, v in sys.modules.items() if owned(k)}
    for name in previous:
        del sys.modules[name]
    old_path = list(sys.path)
    sys.path.insert(0, str(VENDOR))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(previous)
        sys.path[:] = old_path


def kl(p, q):
    return torch.mean(torch.sum(p * (torch.log(p + .0001) - torch.log(q + .0001)), dim=-1), dim=1)


def associations(series, priors):
    series_loss, prior_loss = 0., 0.
    for s, p in zip(series, priors):
        p = p / p.sum(dim=-1, keepdim=True)
        series_loss += kl(s, p.detach()).mean() + kl(p.detach(), s).mean()
        prior_loss += kl(p, s.detach()).mean() + kl(s.detach(), p).mean()
    return series_loss / len(priors), prior_loss / len(priors)


class FittedFCM:
    def __init__(self, model, scaler, config, metadata):
        self.model, self.scaler, self.config, self.metadata = model, scaler, config, metadata

    def transform(self, x):
        return self.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    def score(self, windows):
        x = self.transform(windows)
        model, cfg = self.model, self.config
        result = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x), cfg["batch_size"]):
                history = torch.from_numpy(x[start:start + cfg["batch_size"]]).to(cfg["device"])
                future = model(history, None, None, None)[0]
                _, reconstruction, series, priors, _ = model(future.detach(), None, history, None, con_cat=True)
                series_loss, prior_loss = 0., 0.
                for s, p in zip(series, priors):
                    p = p / p.sum(dim=-1, keepdim=True)
                    series_loss += kl(s, p.detach()) * cfg["temperature"]
                    prior_loss += kl(p, s.detach()) * cfg["temperature"]
                energy = torch.softmax(-series_loss - prior_loss, dim=-1) * ((history - reconstruction) ** 2).mean(dim=-1)
                result.append(energy.sum(dim=-1).cpu().numpy())
        return np.concatenate(result).astype(float) if result else np.empty(0)


def fit(train, future, validation, validation_future, output: Path, *, batch_size=8, seed=20261110, epochs=20, device="cuda"):
    global _MODEL
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    if _MODEL is None:
        with author_imports():
            _MODEL = importlib.import_module("models.FCM").Model
    cfg = dict(seed=seed, epochs=epochs, batch_size=batch_size, device=device, learning_rate=.001,
               d_model=128, d_ff=128, e_layers=4, n_heads=8, dropout=.1, factor=3,
               k=3., temperature=50., context_start_epoch=1, score_reduction="sum")
    args = SimpleNamespace(task_name="long_term_forecast", seq_len=train.shape[1], pred_len=train.shape[1],
                           enc_in=train.shape[2], c_out=train.shape[2], d_model=128, d_ff=128,
                           e_layers=4, n_heads=8, dropout=.1, factor=3, output_attention=False,
                           embed="timeF", freq="d", activation="gelu")
    scaler = StandardScaler().fit(train.reshape(-1, train.shape[-1]))
    def tensor(x):
        value = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
        return torch.from_numpy(value).to(device)
    x, y, vx, vy = [tensor(a) for a in (train, future, validation, validation_future)]
    model = _MODEL(args).float().to(device)
    initial_context = model.con_cat.weight.detach().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(seed)
    history, best, best_state, best_epoch = [], float("inf"), None, None
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); losses = []
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            ids = torch.as_tensor(order[start:start + batch_size], device=device)
            bx, by = x[ids], y[ids]
            optimizer.zero_grad(set_to_none=True)
            prediction, reconstruction, series, priors, _ = model(bx, None, None, None)
            if epoch >= cfg["context_start_epoch"]:
                _, reconstruction, series, priors, _ = model(prediction.detach(), None, bx, None, con_cat=True)
            series_loss, prior_loss = associations(series, priors)
            reconstruction_loss = ((reconstruction - bx) ** 2).mean()
            forecast_loss = ((prediction - by) ** 2).mean()
            loss = 2 * reconstruction_loss - cfg["k"] * series_loss + cfg["k"] * prior_loss + forecast_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"FCM nonfinite loss at epoch {epoch+1}")
            loss.backward(); optimizer.step()
            losses.append([float(forecast_loss.detach()), float(reconstruction_loss.detach())])
        model.eval(); validation_sum = 0.
        with torch.no_grad():
            for start in range(0, len(vx), batch_size):
                prediction = model(vx[start:start + batch_size], None, None, None)[0]
                validation_sum += float(((prediction - vy[start:start + batch_size]) ** 2).mean()) * len(prediction)
        validation_loss = validation_sum / len(vx)
        if epoch >= cfg["context_start_epoch"] and validation_loss < best:
            best, best_epoch = validation_loss, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        means = np.mean(losses, axis=0)
        history.append(dict(epoch=epoch+1, forecast_mse=float(means[0]), reconstruction_mse=float(means[1]),
                            validation_forecast_mse=validation_loss, seconds=time.perf_counter()-started))
        print(f"FCM {output.parent.parent.name} {output.name} epoch {epoch+1}/{epochs} val_mse={validation_loss:.6g}", flush=True)
        for group in optimizer.param_groups: group["lr"] = cfg["learning_rate"] * .5 ** epoch
    model.load_state_dict(best_state)
    metadata = dict(method="FCM", config=cfg, model_config=vars(args), epochs_completed=len(history), best_epoch=best_epoch,
                    train_windows=len(train), validation_windows=len(validation), history=history,
                    parameter_count=sum(p.numel() for p in model.parameters()), training_seconds=time.perf_counter()-started,
                    selected_context_weight_max_change=float((model.con_cat.weight.detach().cpu()-initial_context).abs().max()),
                    checkpoint_selection="minimum validation forecast MSE after context activation; all20 epochs run",
                    score="author KL-weighted history reconstruction energy summed over32 days; predicted future only at inference",
                    adaptations=["daily32x11 data loader", "existing device/autograd compatibility patches",
                                 "remove upstream context no_grad; preserve detached prediction", "no future supplied at held-person inference"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, config=cfg, model_config=vars(args), scaler_mean=scaler.mean_, scaler_scale=scaler.scale_), output/"checkpoint.pt")
    return FittedFCM(model, scaler, cfg, metadata)
