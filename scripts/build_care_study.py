"""CARE native fault endpoint and source-free, fixed-budget HNE features.

The endpoint is the official fault boundary, not a physiological fluctuation label.
Only the first 56 days determine normalization and channel selection. No target
classifier is fitted, and official annotations never enter the feature encoder.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.daily_protocol import robust_reference, normalize
from src.hne_representation import Sampling, encode_person, select_reference_channels
from src.dnb import score_window


def config():
    return {
        "version": "care-native-fault-transfer-v1-20261110",
        "endpoint": "official_event_end_for_anomaly_cases_only",
        "reporting": "RQ3 native industrial disruption; distinct from human future fluctuation endpoint",
        "reference_days": 56, "minimum_reference_observations": 28, "mad_multiplier": 1.4826,
        "reference_scale": "1.4826*MAD; fallback population SD; constant/unsupported coordinates ineligible",
        "history_days": 32, "minimum_origin_index_zero_based": 87,
        "issue_time": "daily available_at[d], the midnight after calendar day d",
        "history_index_interval": "[d-31,d+1); exclusive cutoff=d+1; reference=[0,56)",
        "issue_eligibility": "prediction_start <= issue < prediction_last_timestamp+10min and issue < fault_time if known",
        "prediction_interval": "[first published prediction timestamp,last published prediction timestamp+10min)",
        "forecast_gap_days": 1, "forecast_horizon_days": 7,
        "future_datetime_interval": "[issue+1 day,issue+8 days)",
        "positive": "anomaly case and issue+1d <= official event_end < issue+8d; retain known failures despite truncated later follow-up",
        "negative": "no fault in future interval and future interval right endpoint <= prediction followup end",
        "exclude_gap_failure": "issue <= known fault < issue+1d",
        "exclude_post_failure": "issue >= known fault",
        "right_censoring": "unobserved full negative horizon excluded; internal SCADA gaps retained as missing measurements and recorded separately",
        "normal_event_end": "designated normal interval end, never interpreted as a failure",
        "official_event_start": "retrospective interval annotation; not used for labeling or feature extraction",
        "past_block_requirement": "none; no human >=2-family rule imposed on industrial labels",
        "state_coordinate_budget": 32,
        "channel_selection": "reference-only; first maximum coverage, then maximum coverage*(1-max absolute paired reference Pearson correlation to selected); tie by field name",
        "channel_selection_implementation": "src.hne_representation.select_reference_channels; input is prefix-normalized values",
        "coverage_denominator_days": 56, "reference_pair_minimum": 12,
        "undefined_pair_correlation": "maximal similarity=1, so unknown paired correlation is not rewarded as novelty",
        "context_selection": "retain every documented environment coordinate; normalize using same first56-day reference",
        "clustering": "farm-specific asset_group; cases never concatenated",
        "normalization_and_encoder_fit_labels": False,
        "source_model_fit_or_target_model_selection": False,
        "dnb": "unchanged src.dnb.score_window on selected normalized past32 states; minimum=12; native score direction, no target fit",
        "sampling": Sampling.daily().__dict__,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def label_origin(issue, prediction_start, followup_end, fault, cfg):
    """Half-open datetime endpoints; positive failures do not require later survival."""
    future_start = issue + np.timedelta64(cfg["forecast_gap_days"], "D")
    future_end = future_start + np.timedelta64(cfg["forecast_horizon_days"], "D")
    if issue < prediction_start or issue >= followup_end:
        return None, "outside_prediction_interval", future_start, future_end
    if not np.isnat(fault) and issue >= fault:
        return None, "at_or_after_fault", future_start, future_end
    if not np.isnat(fault) and issue <= fault < future_start:
        return None, "fault_in_gap", future_start, future_end
    if not np.isnat(fault) and future_start <= fault < future_end:
        return 1, "known_future_fault", future_start, future_end
    if future_end <= followup_end:
        return 0, "complete_negative_followup", future_start, future_end
    return None, "negative_right_censored", future_start, future_end


def process_case(task):
    source, output, event, cfg = task
    source, output = Path(source), Path(output)
    case = event["case_id"]
    with np.load(source / "daily" / f"{case}.npz", allow_pickle=False) as z:
        values = z["states"].astype(np.float64)
        context = z["contexts"].astype(np.float64)
        names, context_names = z["state_names"], z["context_names"]
        available = z["available_at"].astype("datetime64[s]")
        dates = z["dates"].astype("datetime64[D]")
        asset = str(z["asset_group"])
    if not np.all(available == dates.astype("datetime64[s]") + np.timedelta64(1, "D")):
        raise ValueError(f"Unexpected daily availability: {case}")
    if not np.all(np.diff(available) == np.timedelta64(1, "D")):
        raise ValueError(f"Incomplete daily calendar: {case}")
    annotations = pd.read_csv(source / "annotations" / f"{case}_timestamps.csv.gz",
                              usecols=["time_stamp", "id", "train_test"])
    timestamps = pd.to_datetime(annotations.time_stamp).to_numpy(dtype="datetime64[s]")
    if not np.all(np.diff(timestamps) > np.timedelta64(0, "s")):
        raise ValueError(f"Annotation clock is not strictly ordered: {case}")
    prediction = timestamps[annotations.train_test.eq("prediction")]
    prediction_start, followup_end = prediction.min(), prediction.max() + np.timedelta64(10, "m")
    training = timestamps[annotations.train_test.eq("train")]
    if len(training) and training.max() >= prediction_start:
        raise ValueError(f"Published training/prediction segments overlap: {case}")
    end_row = annotations.id.eq(int(event["event_end_id"]))
    event_end = np.datetime64(event["event_end"], "s")
    if end_row.sum() != 1 or timestamps[end_row.to_numpy()][0] != event_end:
        raise ValueError(f"Official event_end does not match its source row ID: {case}")
    if annotations.loc[end_row, "train_test"].iloc[0] != "prediction":
        raise ValueError(f"Official event_end is not in the published prediction segment: {case}")
    fault = event_end if event["event_label"] == "anomaly" else np.datetime64("NaT", "s")
    with threadpool_limits(limits=1):
        center, scale, count = robust_reference(values, cfg["reference_days"],
            cfg["minimum_reference_observations"], cfg["mad_multiplier"])
        all_normalized = normalize(values, center, scale)
        selected = select_reference_channels(all_normalized, names, reference=cfg["reference_days"],
            budget=cfg["state_coordinate_budget"], minimum=cfg["minimum_reference_observations"])
        selection = [{"order": k, "source_index": int(j), "field": str(names[j]),
            "reference_observations": int(count[j]), "coverage": float(count[j]/cfg["reference_days"])}
            for k, j in enumerate(selected)]
        ccenter, cscale, ccount = robust_reference(context, cfg["reference_days"],
            cfg["minimum_reference_observations"], cfg["mad_multiplier"])
        normalized = all_normalized[:, selected]
        normalized_context = normalize(context, ccenter, cscale)
    rows = []
    for d in range(cfg["minimum_origin_index_zero_based"], len(values)):
        issue = available[d]
        if not (prediction_start <= issue < followup_end):
            continue
        history_start = d - cfg["history_days"] + 1
        if history_start < cfg["reference_days"] or available[history_start:d+1].max() != issue:
            raise ValueError(f"History/reference boundary violation: {case}/{d}")
        y, reason, future_start, future_end = label_origin(issue, prediction_start, followup_end, fault, cfg)
        if not len(selected):
            y, reason = None, "no_eligible_reference_state"
        # Count missing source slots only over the portion actually followed up.
        observed_end = min(future_end, followup_end)
        expected = max(0, int((observed_end - future_start) / np.timedelta64(10, "m")))
        observed = int(np.count_nonzero((prediction >= future_start) & (prediction < observed_end)))
        rows.append({"dataset": "CARE-v6", "case_id": case, "asset_group": asset,
            "person_id": asset, "farm": event["farm"], "event_id": int(event["event_id"]),
            "origin_day": d, "history_start_day": history_start, "cutoff": d+1,
            "origin_date": str(dates[d]), "issue_time": str(issue),
            "future_start": str(future_start), "future_end_exclusive": str(future_end),
            "prediction_start": str(prediction_start), "prediction_end_exclusive": str(followup_end),
            "fault_time": "" if np.isnat(fault) else str(fault),
            "y": np.nan if y is None else y, "eligible_sample": y is not None, "reason": reason,
            "future_complete_followup": bool(future_end <= followup_end),
            "future_missing_10min_slots_before_censoring": max(0, expected-observed),
            "selected_state_count": len(selected), "context_count": context.shape[1]})
    accepted = [row for row in rows if row["eligible_sample"]]
    cutoff = np.asarray([row["cutoff"] for row in accepted], dtype=np.int32)
    dnb = np.asarray([score_window(normalized[t-cfg["history_days"]:t], minimum=12) for t in cutoff])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        if len(cutoff):
            # Do not even pass observations after the last eligible issue to HNE.
            nodes, pooled, groups = encode_person(normalized[:cutoff.max()],
                normalized_context[:cutoff.max()], cutoff, Sampling.daily())
        else:
            nodes = np.empty((0, len(selected), 50), np.float32)
            pooled = np.empty((0, 296), np.float32)
            groups = np.asarray(["H"]*96 + ["N"]*36 + ["E"]*128 + ["NE"]*36)
    if pooled.shape != (len(accepted), 296) or nodes.shape != (len(accepted), len(selected), 50):
        raise ValueError(f"Representation schema changed: {case}")
    np.save(output / "nodes" / f"{case}.npy", nodes)
    np.savez_compressed(output / "cases" / f"{case}.npz", normalized=normalized.astype(np.float32),
        normalized_contexts=normalized_context.astype(np.float32), available_at=available, dates=dates,
        state_names=names[selected], selected_names=names[selected], context_names=context_names, selected_source_indices=selected,
        reference_location=center[selected], reference_scale=scale[selected],
        reference_observation_count=count[selected], context_reference_location=ccenter,
        context_reference_scale=cscale, context_reference_observation_count=ccount,
        pooled=pooled, groups=groups, cutoffs=cutoff, dnb_score=dnb,
        asset_group=np.asarray(asset), case_id=np.asarray(case))
    write_json(output / "selection" / f"{case}.json", {"case_id": case, "asset_group": asset,
        "source_state_count": len(names), "eligible_reference_states": int(np.isfinite(scale).sum()),
        "selected": selection, "all_context_fields": context_names.tolist()})
    for k, row in enumerate(accepted):
        row["case_node_row"] = k
        row["dnb_score"] = float(dnb[k])
    summary = {"case_id": case, "asset_group": asset, "farm": event["farm"],
        "official_event_label": event["event_label"], "prediction_origins": len(rows),
        "samples": len(accepted), "positive": sum(r["y"] == 1 for r in accepted),
        "negative": sum(r["y"] == 0 for r in accepted), "selected_state_count": len(selected),
        "supported_context_count": int(np.isfinite(cscale).sum()),
        "dnb_valid": int(np.isfinite(dnb).sum()),
        "positive_without_complete_future_followup": sum(r["y"] == 1 and not r["future_complete_followup"] for r in accepted)}
    return rows, accepted, summary, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/care_transfer")
    parser.add_argument("--output", type=Path, default=ROOT / "data/care_study")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cfg = config()
    for folder in ("cases", "selection", "nodes"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", cfg)
    events = pd.read_csv(args.source / "annotations/events.csv").sort_values(["farm", "event_id"])
    tasks = [(str(args.source), str(args.output), event, cfg) for event in events.to_dict("records")]
    all_rows, accepted, summaries, pooled = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for rows, valid, summary, features in executor.map(process_case, tasks):
            all_rows.extend(rows); accepted.extend(valid); summaries.append(summary); pooled.append(features)
            print(json.dumps(summary), flush=True)
    origins, samples, cases = pd.DataFrame(all_rows), pd.DataFrame(accepted), pd.DataFrame(summaries)
    samples.insert(0, "sample_id", np.arange(len(samples)))
    origins.to_csv(args.output / "origins.csv", index=False)
    samples.to_csv(args.output / "samples.csv", index=False)
    cases.to_csv(args.output / "cases.csv", index=False)
    groups = np.asarray(["H"]*96 + ["N"]*36 + ["E"]*128 + ["NE"]*36)
    np.savez_compressed(args.output / "features.npz", pooled=np.concatenate(pooled), groups=groups,
        y=samples.y.to_numpy(np.int8), sample_id=samples.sample_id.to_numpy(np.int32),
        dnb_score=samples.dnb_score.to_numpy(np.float64),
        case_id=samples.case_id.to_numpy(dtype=str), asset_group=samples.asset_group.to_numpy(dtype=str),
        person_id=samples.asset_group.to_numpy(dtype=str), farm=samples.farm.to_numpy(dtype=str),
        origin_day=samples.origin_day.to_numpy(np.int32), case_node_row=samples.case_node_row.to_numpy(np.int32),
        issue_time=pd.to_datetime(samples.issue_time).to_numpy(dtype="datetime64[s]"),
        future_start=pd.to_datetime(samples.future_start).to_numpy(dtype="datetime64[s]"),
        future_end_exclusive=pd.to_datetime(samples.future_end_exclusive).to_numpy(dtype="datetime64[s]"))
    by_farm = {farm: {"samples": len(f), "positive": int(f.y.sum()), "negative": int((f.y == 0).sum()),
               "scorable_cases": f.case_id.nunique(), "scorable_assets": f.asset_group.nunique()}
               for farm, f in samples.groupby("farm")}
    summary = {"samples": len(samples), "positive": int(samples.y.sum()), "negative": int((samples.y == 0).sum()),
        "scorable_cases": samples.case_id.nunique(), "scorable_assets": samples.asset_group.nunique(),
        "positive_cases": samples.loc[samples.y == 1, "case_id"].nunique(),
        "positive_assets": samples.loc[samples.y == 1, "asset_group"].nunique(),
        "cases_without_samples": cases.loc[cases.samples == 0, "case_id"].tolist(),
        "by_farm": by_farm, "origin_reasons": origins.reason.value_counts().to_dict(),
        "positive_without_complete_future_followup": int(cases.positive_without_complete_future_followup.sum()),
        "negative_with_internal_scada_gaps": int(((samples.y == 0) & (samples.future_missing_10min_slots_before_censoring > 0)).sum()),
        "dnb_valid": int(np.isfinite(samples.dnb_score).sum()),
        "representation": "unchanged Sampling.daily/encode_person; pooled296 H96/N36/E128/NE36; variable nodes50 by case",
        "config_sha256": hashlib.sha256((args.output / "config.json").read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256((args.source / "manifest.json").read_bytes()).hexdigest(),
        "model_trained": False, "target_metric_computed": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
