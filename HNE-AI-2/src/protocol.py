"""One fixed personal-reference future-block definition for all daily cohorts."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DailyReference:
    location: np.ndarray
    scale: np.ndarray
    observation_count: np.ndarray
    reference_block_count: np.ndarray
    amplitude_threshold: np.ndarray
    bias_threshold: np.ndarray
    absolute_point_threshold: np.ndarray
    eligible: np.ndarray
    families: np.ndarray
    family_names: np.ndarray
    eligible_families: np.ndarray


def robust_reference(values: np.ndarray, reference_days: int, minimum: int,
                     mad_multiplier: float = 1.4826):
    """Prefix-only scale. No fill, clipping, future data, or constant-column scale."""
    values = np.asarray(values, dtype=np.float64)
    center = np.full(values.shape[1], np.nan)
    scale = np.full(values.shape[1], np.nan)
    counts = np.isfinite(values[:reference_days]).sum(axis=0)
    for j in range(values.shape[1]):
        x = values[:reference_days, j]
        x = x[np.isfinite(x)]
        if len(x) < minimum:
            continue
        center[j] = np.median(x)
        if np.ptp(x) == 0:
            continue
        candidate = mad_multiplier * np.median(np.abs(x - center[j]))
        if candidate <= 0:
            candidate = np.std(x, ddof=0)
        if candidate > 0 and np.isfinite(candidate):
            scale[j] = candidate
    return center, scale, counts


def normalize(values: np.ndarray, location: np.ndarray, scale: np.ndarray):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.asarray(values, dtype=np.float64) - location) / scale


def fit_reference(states: np.ndarray, families: list[str] | np.ndarray, config: dict):
    """Fit all event thresholds exclusively from each person's initial 56 days."""
    states = np.asarray(states, dtype=np.float64)
    families = np.asarray(families, dtype=str)
    if states.ndim != 2 or len(families) != states.shape[1]:
        raise ValueError("State columns and family labels must agree")
    r, w = config["reference_days"], config["block_days"]
    center, scale, count = robust_reference(states, r, config["minimum_reference_observations"], config["mad_multiplier"])
    z = normalize(states, center, scale)
    n_features = states.shape[1]
    amp_cut = np.full(n_features, np.nan)
    bias_cut = np.full(n_features, np.nan)
    point_cut = np.full(n_features, np.nan)
    block_count = np.zeros(n_features, dtype=np.int32)
    qlow, qhigh = config["amplitude_quantiles"]
    for j in range(n_features):
        if not np.isfinite(scale[j]) or len(states) < r:
            continue
        reference = z[:r, j]
        finite = reference[np.isfinite(reference)]
        point_cut[j] = np.quantile(np.abs(finite), config["amplitude_point_abs_z_quantile"], method="linear")
        amps, biases = [], []
        for start in range(r - w + 1):
            block = reference[start:start + w]
            block = block[np.isfinite(block)]
            if len(block) < config["minimum_block_observations"]:
                continue
            amps.append(np.quantile(block, qhigh, method="linear") - np.quantile(block, qlow, method="linear"))
            biases.append(abs(np.mean(block)))
        block_count[j] = len(amps)
        if len(amps) < config["minimum_reference_blocks"]:
            continue
        amp_cut[j] = max(np.quantile(amps, config["reference_block_quantile"], method="linear"),
                         config["amplitude_reference_median_multiplier"] * np.median(amps))
        bias_cut[j] = max(np.quantile(biases, config["reference_block_quantile"], method="linear"),
                          config["bias_floor_z"])
    eligible = np.isfinite(amp_cut) & np.isfinite(bias_cut) & np.isfinite(point_cut)
    names = np.asarray(sorted(set(families)), dtype=str)
    eligible_families = np.asarray([np.any(eligible & (families == name)) for name in names])
    reference = DailyReference(center, scale, count, block_count, amp_cut, bias_cut, point_cut,
                               eligible, families, names, eligible_families)
    return reference, z


def label_blocks(normalized: np.ndarray, reference: DailyReference, config: dict):
    """Three-valued labels for seven-day blocks indexed by their final day.

    A variable's unknown status is -1; a block's unknown label is NaN.
    Families that lacked sufficient reference data are excluded rather than
    pretending to support a negative label.
    """
    z = np.asarray(normalized, dtype=np.float64)
    n, p = z.shape
    w = config["block_days"]
    variable = np.full((n, p), -1, dtype=np.int8)
    observation_count = np.zeros((n, p), dtype=np.int16)
    amplitude = np.full((n, p), np.nan)
    bias = np.full((n, p), np.nan)
    extreme_count = np.zeros((n, p), dtype=np.int8)
    same_sign_count = np.zeros((n, p), dtype=np.int8)
    low, high = config["amplitude_quantiles"]
    for end in range(w - 1, n):
        for j in range(p):
            block = z[end - w + 1:end + 1, j]
            x = block[np.isfinite(block)]
            observation_count[end, j] = len(x)
            if not reference.eligible[j] or len(x) < config["minimum_block_observations"]:
                continue
            a = np.quantile(x, high, method="linear") - np.quantile(x, low, method="linear")
            b = abs(np.mean(x))
            extreme = np.count_nonzero(np.abs(x) > reference.absolute_point_threshold[j])
            same_sign = max(np.count_nonzero(x > config["bias_same_sign_z"]),
                            np.count_nonzero(x < -config["bias_same_sign_z"]))
            amplitude[end, j], bias[end, j] = a, b
            extreme_count[end, j], same_sign_count[end, j] = extreme, same_sign
            amp_abnormal = a > reference.amplitude_threshold[j] and extreme >= config["amplitude_minimum_extreme_days"]
            bias_abnormal = b > reference.bias_threshold[j] and same_sign >= config["bias_minimum_same_sign_days"]
            variable[end, j] = int(amp_abnormal or bias_abnormal)
    family = np.full((n, len(reference.family_names)), -1, dtype=np.int8)
    for k, name in enumerate(reference.family_names):
        selected = reference.eligible & (reference.families == name)
        if not np.any(selected):
            continue
        v = variable[:, selected]
        family[np.all(v == 0, axis=1), k] = 0
        family[np.any(v == 1, axis=1), k] = 1
    active = family[:, reference.eligible_families]
    abnormal = (active == 1).sum(axis=1)
    unknown = (active == -1).sum(axis=1)
    required = config["minimum_abnormal_families"]
    labels = np.full(n, np.nan)
    if reference.eligible_families.sum() >= required:
        labels[abnormal >= required] = 1.
        labels[abnormal + unknown < required] = 0.
    return {"label": labels, "variable_status": variable, "family_status": family,
            "variable_observation_count": observation_count, "amplitude": amplitude, "bias": bias,
            "extreme_day_count": extreme_count, "same_sign_day_count": same_sign_count,
            "known_variables": (variable != -1).sum(axis=1), "abnormal_families": abnormal,
            "unknown_families": unknown, "known_families": (active != -1).sum(axis=1)}


def forecast_origins(blocks: dict, available_at: np.ndarray, dates: np.ndarray, config: dict):
    """Require a normal past block and issue after all history indices are available."""
    n = len(blocks["label"])
    availability = np.asarray(available_at, dtype="datetime64[s]")
    calendar_starts = np.asarray(dates, dtype="datetime64[D]").astype("datetime64[s]")
    future = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    issues = np.full(n, np.datetime64("NaT", "s"), dtype="datetime64[s]")
    rows = []
    start = config["minimum_origin_index_zero_based"]
    end_offset = config["future_end_offset"]
    for d in range(start, n - end_offset):
        history_start = d - config["history_days"] + 1
        if history_start < config["reference_days"]:
            raise ValueError("Monitoring history overlaps the reference prefix")
        history_available = availability[history_start:d + 1]
        if np.isnat(history_available).any():
            continue
        issue = history_available.max()
        # Anchor to the target period's start, not its later reporting timestamp.
        # A complete one-day gap must remain after all history data are available.
        latest_issue = calendar_starts[d + config["future_start_offset"]] - np.timedelta64(config["forecast_gap_days"], "D")
        if issue > latest_issue:
            continue
        target_end = d + end_offset
        future[d] = blocks["label"][target_end]
        issues[d] = issue
        accepted[d] = blocks["label"][d] == 0 and np.isfinite(future[d])
        rows.append({"origin_day": d, "history_start_day": history_start,
                     "future_start_day": d + config["future_start_offset"], "future_end_day": target_end,
                     "issue_time": str(issue), "past_block_label": blocks["label"][d], "y": future[d],
                     "eligible_sample": bool(accepted[d]),
                     "future_known_variables": int(blocks["known_variables"][target_end]),
                     "future_known_families": int(blocks["known_families"][target_end]),
                     "future_abnormal_families": int(blocks["abnormal_families"][target_end]),
                     "future_unknown_families": int(blocks["unknown_families"][target_end])})
    return {"future_labels": future, "eligible_origin": accepted, "issue_time": issues, "origin_rows": rows}
