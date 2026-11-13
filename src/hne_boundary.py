"""Past-only distances to the existing personal sustained-disturbance boundary."""
import numpy as np


def boundary_coordinates(data, origin_days, lags=(0, 7, 14, 21)):
    names, families = data["family_names"], data["state_families"]
    eligible = data["reference_eligible"].astype(bool)
    amplitude_cut = data["reference_amplitude_threshold"]
    bias_cut = data["reference_bias_threshold"]
    output = np.full((len(origin_days), len(names), len(lags)*5), np.nan, dtype=np.float32)
    for k, lag in enumerate(lags):
        indexes = np.asarray(origin_days, dtype=int)-lag
        status = data["block_variable_status"][indexes]
        observed = (status >= 0) & eligible[None, :]
        amplitude = np.minimum((data["block_amplitude"][indexes]-amplitude_cut)/(1+amplitude_cut),
                               (data["block_extreme_day_count"][indexes]-2+.5)/2)
        bias = np.minimum((data["block_bias"][indexes]-bias_cut)/(1+bias_cut),
                          (data["block_same_sign_day_count"][indexes]-3+.5)/3)
        amplitude, bias = np.where(observed, amplitude, np.nan), np.where(observed, bias, np.nan)
        union = np.maximum(amplitude, bias)
        for f, name in enumerate(names):
            selected = (families == name) & eligible
            if not selected.any():
                continue
            finite = observed[:, selected]
            count = finite.sum(1)
            for j, values in enumerate([amplitude, bias, union]):
                maximum = np.max(np.where(finite, values[:, selected], -np.inf), axis=1)
                output[:, f, 5*k+j] = np.where(count > 0, maximum, np.nan)
            output[:, f, 5*k+3] = np.divide(((status[:, selected] == 1) & finite).sum(1), count,
                                            out=np.full(len(count), np.nan), where=count > 0)
            output[:, f, 5*k+4] = count/selected.sum()
    return output
