"""Preserve personal-reference units in the existing temporal training loop.

The factory adapter reuses fit_predict's original code object with a private
copy of its globals. Only the training constructor changes; neither the source
module's globals nor the original architecture/optimizer/selection code change.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from . import hne_temporal as temporal


class ReferenceUnitsHNE(temporal.HNETemporal):
    def fit_scalers(self, windows, nodes, pooled, origin_weights):
        super().fit_scalers(windows, nodes, pooled, origin_weights)
        p = nodes.shape[1]
        self.input_scaling = {"partition": "train", "origins": len(windows),
            "diagnostic": "training-only compression of personal-reference units by the original global scale",
            "quantiles": [.5, .9, .99], "groups": []}
        for prefix, block in [("state", windows[:, :, :p]), ("context", windows[:, :, p:])]:
            original_mean = float(getattr(self, prefix+"_mean").item())
            original_scale = float(getattr(self, prefix+"_scale").item())
            x = block[np.isfinite(block)].astype(np.float64)
            original_input = np.clip((x-original_mean)/original_scale, -8., 8.)
            original_zero = np.clip(-original_mean/original_scale, -8., 8.)
            reference_input = np.clip(x, -8., 8.)
            self.input_scaling["groups"].append({"group": prefix, "observed": len(x),
                "original_train_mean": original_mean, "original_train_scale": original_scale,
                "reference_units_abs_quantile": np.quantile(np.abs(x), [.5,.9,.99]).tolist() if len(x) else None,
                "original_fraction_within_hundredth_of_zero_input": float(np.mean(np.abs(original_input-original_zero)<.01)) if len(x) else None,
                "reference_units_fraction_within_hundredth_of_zero_input": float(np.mean(np.abs(reference_input)<.01)) if len(x) else None,
                "new_mean": 0., "new_scale": 1.,
                "supported": bool(getattr(self, prefix+"_supported").item())})
            getattr(self, prefix+"_mean").zero_()
            getattr(self, prefix+"_scale").fill_(1.)
        # node/relation buffers and all supported masks remain exactly as fitted.


def fit_predict(train, validation, test, groups, seed, learning_rate, output,
                device="auto", run_fingerprint=None):
    created = []

    def factory(*args, **kwargs):
        model = ReferenceUnitsHNE(*args, **kwargs)
        created.append(model)
        return model

    original = temporal.fit_predict
    implementation = FunctionType(original.__code__, {**original.__globals__, "HNETemporal": factory},
                                  original.__name__, original.__defaults__, original.__closure__)
    implementation.__kwdefaults__ = original.__kwdefaults__
    result = implementation(train, validation, test, groups, seed, learning_rate, output,
                            device=device, run_fingerprint=run_fingerprint)
    metadata = result["metadata"]
    metadata.update(model="HNE-reference-units",
        normalization="personal-reference state/context units preserved with mean0/scale1; original clip±8 and observed masks; node50 and relation72 training scalers unchanged",
        input_scaling=created[0].input_scaling)
    if output is not None:
        folder = Path(output)
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        checkpoint["metadata"] = metadata
        torch.save(checkpoint, folder / "model.pt")
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        (folder / "input_scaling.json").write_text(json.dumps(metadata["input_scaling"], indent=2, allow_nan=False), encoding="utf-8")
    return result
