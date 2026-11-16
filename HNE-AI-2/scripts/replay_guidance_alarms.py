"""Replay all history-eligible issue days; fit alarm thresholds on Bris only."""
import hashlib
import json
import os
from pathlib import Path
import time

from run_guidance_transfer import ROOT, forecast, freeze
import numpy as np
import pandas as pd
import psutil
import torch
from fusion.evidence import FullEvidenceFusion
from precursor_guidance import ConditionalPredictor, EvidenceBias
from train_backbone import load_adapter, load_backbone
from train_comparison import resource_check, save_json, sha
from train_evidence import calibrate_person
from training_data import SOURCE, load_people


def schedule(dataset, fold, protocol):
    rows = []
    for file in sorted((SOURCE / 'data/daily_study' / dataset).glob('*_daily.npz')):
        with np.load(file, allow_pickle=False) as a:
            if int(a['fold']) != fold:
                continue
            dates, issued = a['dates'], a['issue_time']
            first, last = protocol['future_start_offset'], protocol['future_end_offset']
            # issue_time uses only history availability. Horizon end limits the replay calendar.
            days = np.arange(protocol['minimum_origin_index_zero_based'], len(dates)-last)
            days = days[(a['block_label'][days]==0) & ~np.isnat(issued[days])]
            if not len(days):
                continue
            hourly = SOURCE / 'data/development' / dataset / file.name.replace('_daily', '_hourly')
            with np.load(hourly, allow_pickle=False) as raw:
                clock = pd.Timestamp(str(raw['clock']))
            if clock.tzinfo is not None:
                clock = clock.tz_convert('UTC').tz_localize(None)
            ends = clock + pd.to_timedelta(days + 1, unit='D')
            if (ends > pd.to_datetime(issued[days])).any():
                raise ValueError('A model input hour extends past issuance')
            labels = a['future_labels'][days]
            for d,y in zip(days,labels):
                start = pd.Timestamp(dates[d+first])
                issue = pd.Timestamp(issued[d])
                if start-issue < pd.Timedelta(days=protocol['forecast_gap_days']):
                    raise ValueError('Insufficient gap at issuance')
                rows.append(dict(person=str(a['person_id']), origin=int(d), issue=issue,
                    target_start=start, target_end=pd.Timestamp(dates[d+last]), label=float(y)))
    frame = pd.DataFrame(rows).sort_values(['person','issue']).reset_index(drop=True)
    if frame.duplicated(['person','issue']).any():
        raise ValueError('More than one issue per person and time')
    return frame


def chronological_alerts(rows, method, threshold, silence_days):
    emitted = []
    previous = {}
    # No future label or target-window column is accessible to the alarm decision.
    for i, person, issue, score in rows[['person','issue',method]].itertuples():
        if score >= threshold and (person not in previous or issue-previous[person] >= pd.Timedelta(days=silence_days)):
            emitted.append(i)
            previous[person] = issue
    return rows.loc[emitted]


def source_threshold(frame, method, cfg):
    exposure = int(frame.label.notna().sum())
    options = np.r_[np.unique(frame[method]), np.nextafter(frame[method].max(), np.inf)]
    for cutoff in options:
        alerts = chronological_alerts(frame, method, cutoff, cfg['silence_days'])
        false_count = int((alerts.label==0).sum())
        if false_count * 30 <= cfg['false_alarms_per_30_scorable_issue_days'] * exposure:
            return float(cutoff)
    raise RuntimeError('No feasible alarm cutoff')


def score_events(rows, method, threshold, silence):
    alerts = chronological_alerts(rows, method, threshold, silence)
    events = []
    for person, group in rows[rows.label==1].groupby('person'):
        for start, end in group[['target_start','target_end']].sort_values('target_start').itertuples(index=False):
            if events and events[-1]['person']==person and start<=events[-1]['end']+pd.Timedelta(days=1):
                events[-1]['end'] = max(end, events[-1]['end'])
            else:
                events.append(dict(person=person,start=start,end=end))
    for event in events:
        hits = alerts[(alerts.person==event['person']) & (alerts.label==1)
                      & (alerts.target_start>=event['start']) & (alerts.target_end<=event['end'])
                      & (alerts.issue<event['start'])]
        event['first_alert'] = hits.issue.min()
        event['lead_days'] = None if hits.empty else (event['start']-hits.issue.min()).total_seconds()/86400
    leads = [e['lead_days'] for e in events if e['lead_days'] is not None]
    n = int(rows.label.notna().sum())
    false = int((alerts.label==0).sum())
    summary = dict(threshold=threshold, candidate_issue_days=len(rows), scorable_issue_days=n,
        unknown_days=len(rows)-n, alerts=len(alerts), false_alerts=false,
        alerts_unknown=int(alerts.label.isna().sum()), alerts_positive=int((alerts.label==1).sum()),
        false_per_30_scorable_issue_days=30*false/n, timely_hits=len(leads), events=len(events),
        recall=len(leads)/len(events) if events else None,
        median_lead_days=float(np.median(leads)) if leads else None)
    return summary, alerts, events


def main():
    t0 = time.perf_counter()
    transfer = json.loads((ROOT / 'configs/guidance_transfer.json').read_text())
    cfg = json.loads((ROOT / 'configs/alarm_replay.json').read_text())
    protocol = json.loads((ROOT / 'configs/daily_protocol.json').read_text())
    folder = ROOT / 'results/guidance_transfer'
    artifacts = ROOT / 'artifacts/guidance_transfer'
    completed = json.loads((artifacts / 'status.json').read_text())
    if completed['state'] != 'complete':
        raise RuntimeError('Complete the matched source/target forecasts first')
    fold,seed=transfer['outer_fold'],transfer['seed']
    checkpoint = ROOT / 'artifacts/key_comparison' / transfer['source_profile'] / cfg['source'] / f'fold{fold}' / f'seed{seed}'
    datasets = (cfg['source'],cfg['target'])
    signature = dict(config=cfg, transfer=sha(artifacts/'manifest.json'), script=sha(Path(__file__)),
        supplied_scores={d:sha(folder/(d+'_predictions.csv')) for d in datasets},
        protocol=sha(ROOT/'configs/daily_protocol.json'))
    manifest=artifacts/'alarm_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise RuntimeError('Existing replay inputs changed')
    save_json(manifest,signature)
    if (folder/'alarm_summary.json').exists():
        print('Completed alarm replay already present', flush=True)
        return
    torch.set_num_threads(2)
    if os.name=='nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    model,head=load_backbone(SOURCE/'tools/models/chronos-2')
    base=torch.load(checkpoint/'M04_C2_LoRA.pt',weights_only=False)['state']
    load_adapter(model,base['lora'])
    head.load_state_dict(base['head'])
    model.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    guide=freeze(EvidenceBias(),checkpoint/'M10_SAGE_fixed.pt')
    fusion=freeze(FullEvidenceFusion(),checkpoint/'M12_full_evidence_fusion.pt')
    small=[freeze(ConditionalPredictor(),checkpoint/'small'/f'component{i}.pt') for i in range(3)]
    thresholds,summary={},{}
    for dataset in datasets:
        frame=schedule(dataset,(fold+1)%5,protocol)
        supplied=pd.read_csv(folder/(dataset+'_predictions.csv'),dtype={'person':str})
        frame=frame.merge(supplied[['person','origin','label',*cfg['methods']]],
            on=['person','origin'],how='left',validate='one_to_one',suffixes=('','_saved'))
        if not np.array_equal(frame.label.to_numpy(), frame.label_saved.to_numpy(), equal_nan=True) or len(supplied)!=frame.label.notna().sum():
            raise ValueError('Scorable forecasts and issue-day schedule disagree')
        frame=frame.drop(columns='label_saved')
        missing=frame[frame.baseline.isna()]
        people={p.person:p for p in load_people(dataset,fold,only_split='validation')['validation']}
        if not set(missing.person).issubset(people):
            raise ValueError('A history-eligible person lacks an input series; include them before reporting coverage')
        for person_id, group in missing.groupby('person'):
            resource_check('cpu')
            p=people[person_id]
            key=f'{dataset}|{p.person}|{seed}'.encode()
            component=int(hashlib.sha256(key).hexdigest(),16)%len(small)
            cal=calibrate_person(small[component],p,transfer['personal_steps'])
            partial=artifacts/'alarm_forecasts'/dataset/(person_id+'.json')
            if partial.exists():
                forecasts=json.loads(partial.read_text())
            else:
                forecasts={}
                for d in group.origin:
                    scores=forecast(p,int(d),small[component],cal,model,head,guide,fusion)
                    forecasts[str(d)]={m:scores[m] for m in cfg['methods']}
                save_json(partial,forecasts)
            for i,d in group.origin.items():
                for m in cfg['methods']:
                    frame.at[i,m]=forecasts[str(d)][m]
            print(f'{dataset} {person_id}: {len(group)} additional issue days',flush=True)
        if not np.isfinite(frame[cfg['methods']].to_numpy()).all():
            raise ValueError('Missing risk on a prospective issue day')
        frame.to_csv(folder/(dataset+'_issue_days.csv'),index=False)
        summary[dataset]={}
        emitted,events=[],[]
        for method in cfg['methods']:
            if dataset==cfg['source']:
                thresholds[method]=source_threshold(frame,method,cfg)
            result,alerts,observed=score_events(frame,method,thresholds[method],cfg['silence_days'])
            summary[dataset][method]=result
            emitted.append(alerts.assign(method=method))
            events.extend(dict(method=method,**e) for e in observed)
        pd.concat(emitted).to_csv(folder/(dataset+'_alarms.csv'),index=False)
        pd.DataFrame(events).to_csv(folder/(dataset+'_events.csv'),index=False)
    summary['elapsed_seconds']=time.perf_counter()-t0
    save_json(folder/'alarm_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
