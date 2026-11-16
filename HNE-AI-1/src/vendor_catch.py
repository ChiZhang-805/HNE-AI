"""Original CATCH full-FFT/channel-mask reconstruction on daily histories."""
from __future__ import annotations
from pathlib import Path
import random
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

VENDOR = Path(__file__).resolve().parents[1] / "third_party/CATCH"
if str(VENDOR) not in sys.path: sys.path.insert(0, str(VENDOR))
from ts_benchmark.baselines.catch.CATCH import TransformerConfig
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss, frequency_criterion
from ts_benchmark.baselines.catch.utils.tools import adjust_learning_rate


class FittedCATCH:
    def __init__(self, model, scaler, cfg, metadata, device, seed):
        self.model, self.scaler, self.cfg, self.metadata = model, scaler, cfg, metadata
        self.device, self.seed = device, seed

    def score(self, windows):
        x = self.scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape).astype(np.float32)
        results = []
        criterion = frequency_criterion(self.cfg)
        self.model.eval()
        with torch.random.fork_rng(devices=[0] if self.device.startswith("cuda") else []):
            torch.manual_seed(self.seed + 31)
            with torch.no_grad():
                for start in range(0, len(x), self.cfg.batch_size):
                    batch = torch.from_numpy(x[start:start+self.cfg.batch_size]).to(self.device)
                    reconstructed, _, _ = self.model(batch)
                    score = ((batch - reconstructed) ** 2).mean(dim=-1)
                    score += self.cfg.score_lambda * criterion(batch, reconstructed).mean(dim=-1)
                    results.append(score.mean(dim=-1).cpu().numpy())
        return np.concatenate(results).astype(float) if results else np.empty(0)


def fit(train, validation, output: Path, *, batch_size=8, seed=20261110, epochs=20, device="cuda"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    length, channels = train.shape[1:]
    cfg = TransformerConfig(seq_len=length, c_in=channels, enc_in=channels, dec_in=channels, c_out=channels,
                            d_model=64, d_ff=128, e_layers=3, batch_size=batch_size, num_epochs=epochs,
                            patch_size=16, patch_stride=8, inference_patch_size=32, inference_patch_stride=1)
    scaler = StandardScaler().fit(train.reshape(-1, channels))
    def tensor(array):
        return torch.from_numpy(scaler.transform(array.reshape(-1, channels)).reshape(array.shape).astype(np.float32)).to(device)
    x, vx = tensor(train), tensor(validation)
    model = CATCHModel(cfg).to(device)
    initial_mask = model.mask_generator.generator[0].weight.detach().cpu().clone()
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if "mask_generator" not in name], lr=cfg.lr)
    mask_optimizer = torch.optim.Adam(model.mask_generator.parameters(), lr=cfg.Mlr)
    auxiliary = frequency_loss(cfg)
    batch_count = (len(x)+batch_size-1)//batch_size
    mask_interval = max(1, min(batch_count//10, 100))
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch = float("inf"), None, None
    history, mask_steps = [], 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); losses = []
        order = rng.permutation(len(x))
        for batch_index, start in enumerate(range(0, len(x), batch_size)):
            batch = x[torch.as_tensor(order[start:start+batch_size], device=device)]
            optimizer.zero_grad()
            reconstructed, complex_output, discovery = model(batch)
            reconstruction = ((reconstructed-batch)**2).mean()
            frequency = auxiliary(complex_output, model.revin_layer(batch, "transform"))
            loss = reconstruction + cfg.dc_lambda*discovery + cfg.auxi_lambda*frequency
            if not torch.isfinite(loss):
                raise FloatingPointError(f"CATCH nonfinite loss at epoch {epoch+1}")
            # Preserve the released bilevel ordering: apply accumulated mask
            # gradients before the current backward pass, not after it.
            if (batch_index+1)%mask_interval == 0:
                has_gradient = any(p.grad is not None for p in model.mask_generator.parameters())
                mask_optimizer.step(); mask_optimizer.zero_grad()
                mask_steps += int(has_gradient)
            loss.backward(); optimizer.step()
            losses.append([float(reconstruction.detach()), float(frequency.detach()), float(discovery.detach())])
        model.eval(); val_sum = 0.
        with torch.random.fork_rng(devices=[0] if device.startswith("cuda") else []):
            torch.manual_seed(seed+17)
            with torch.no_grad():
                for start in range(0, len(vx), batch_size):
                    batch = vx[start:start+batch_size]
                    reconstruction = model(batch)[0]
                    val_sum += float(((reconstruction-batch)**2).mean())*len(batch)
        validation_loss = val_sum/len(vx)
        if validation_loss < best:
            best, best_epoch = validation_loss, epoch+1
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        values = np.mean(losses, axis=0)
        history.append(dict(epoch=epoch+1, reconstruction_loss=float(values[0]), frequency_loss=float(values[1]),
                            discovery_loss=float(values[2]), validation_reconstruction_mse=validation_loss,
                            main_lr=optimizer.param_groups[0]["lr"], mask_lr=mask_optimizer.param_groups[0]["lr"],
                            seconds=time.perf_counter()-started))
        print(f"CATCH {output.parent.parent.name} {output.name} epoch {epoch+1}/{epochs} val_mse={validation_loss:.6g}", flush=True)
        # The released type1 schedule intentionally uses cfg.lr for both optimizers.
        adjust_learning_rate(optimizer, None, epoch+1, cfg, printout=False)
        adjust_learning_rate(mask_optimizer, None, epoch+1, cfg, printout=False)
    model.load_state_dict(best_state)
    metadata = dict(method="CATCH", config=vars(cfg), seed=seed, device=device, epochs_completed=len(history),
                    train_windows=len(train), validation_windows=len(validation), best_epoch=best_epoch, history=history,
                    parameter_count=sum(p.numel() for p in model.parameters()), training_seconds=time.perf_counter()-started,
                    mask_optimizer_steps=mask_steps,
                    selected_mask_weight_max_change=float((model.mask_generator.generator[0].weight.detach().cpu()-initial_mask).abs().max()),
                    checkpoint_selection="minimum validation reconstruction MSE; all20 epochs run",
                    score="mean over history of author time reconstruction +0.05 frequency criterion",
                    adaptations=["daily32x11 data loader", "d_model64/d_ff128 parameter budget; all FFT/patch/mask/bilevel modules retained",
                                 "existing device/autograd compatibility patches", "fixed validation/score Gumbel streams"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, config=vars(cfg), scaler_mean=scaler.mean_, scaler_scale=scaler.scale_), output/"checkpoint.pt")
    return FittedCATCH(model, scaler, cfg, metadata, device, seed)
