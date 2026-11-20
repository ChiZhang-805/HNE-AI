"""Chronological replay of completed CPU baselines, including unknown outcomes."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from run_local_baselines import ROOT,ART as BASE,OUT as PRED,METHODS,SEEDS,estimator,probability,read_windows,tree_features,resource_check,save_json,sha
from replay_guidance_alarms import schedule,source_threshold,score_events
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from threadpoolctl import threadpool_limits
from study_inputs import SOURCE
OUT=ROOT/'results/local_alarms';ART=ROOT/'artifacts/local_alarms'


def inputs(dataset,fold,protocol):
    frame=schedule(dataset,fold,protocol);people=[]
    paths={}
    for path in (SOURCE/'data/daily_study'/dataset).glob('*_daily.npz'):
        with np.load(path,allow_pickle=False) as raw:paths[str(raw['person_id'])]=path
    for person,part in frame.groupby('person'):
        # read_windows needs an origin schedule; these dummy labels never enter
        # a feature or stored outcome. Real three-valued labels stay in frame.
        people.append(SimpleNamespace(dataset=dataset,person=person,source_files=(str(paths[person]),),
                                      origins=part.origin.to_numpy(),labels=np.zeros(len(part))))
    x,ids=read_windows(people,protocol)
    return frame,x,ids[['dataset','person','origin']]


def main():
    # The imported transfer runner sets '-1'. This Windows XGBoost build
    # segfaults under that mask; empty visibility disables GPUs and loads safely.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu');start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/alarm_replay.json').read_text());protocol=json.loads((ROOT/'configs/daily_protocol.json').read_text())
    provenance=dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/replay_guidance_alarms.py',ROOT/'scripts/run_local_baselines.py',ROOT/'scripts/run_public_baselines.py',ROOT/'src/study_inputs.py',ROOT/'src/training_data.py',ROOT/'configs/daily_protocol.json',ROOT/'configs/alarm_replay.json',BASE/'manifest.json']},
                    scope='Same-cohort outer evaluation folds; thresholds fitted on designated validation people only; unknown outcomes can emit and silence alerts',
                    policy='Lowest validation score satisfying <=1 false alert per30 scorable issue days, with seven-day silence; no-alert candidate included',
                    endpoint='Merged positive target windows; operational lead is structurally one day, not clinical onset lead')
    data_files=[]
    for dataset in ('BrisT1D-Open','RESILIENT'):
        for daily in (SOURCE/'data/daily_study'/dataset).glob('*_daily.npz'):
            data_files.extend([daily,SOURCE/'data/development'/dataset/daily.name.replace('_daily','_hourly')])
    provenance['data_sha256']={str(p):sha(p) for p in sorted(data_files)}
    path=ART/'manifest.json'
    recovery=ROOT/'artifacts/local_model_recovery'
    assert json.loads((recovery/'status.json').read_text())['completed_units']==30
    provenance['model_recovery_manifest_sha256']=sha(recovery/'manifest.json')
    provenance['recovery_receipt_sha256']=sha(ART/'recovery_receipt.json')
    provenance['gpu_visibility']='Empty CUDA_VISIBLE_DEVICES before XGBoost import; CPU models only; -1 reproducibly crashes this local native build'
    if path.exists() and json.loads(path.read_text())!=provenance:
        receipt=json.loads((ART/'recovery_receipt.json').read_text())
        assert sha(path)==receipt['previous_manifest_sha256']
        previous=receipt['previous_manifest'];old_sources=previous['source_sha256'];new_sources=provenance['source_sha256']
        assert set(old_sources)==set(new_sources)
        assert all(old_sources[k]==v for k,v in new_sources.items() if k!=str(Path(__file__).relative_to(ROOT)))
        assert all(provenance[k]==v for k,v in previous.items() if k not in ('source_sha256','recovery_receipt_sha256'))
        for name,expected in receipt['completed_report_sha256'].items():assert sha(ROOT/name)==expected
    save_json(path,provenance);registry=[]
    for dataset in ('BrisT1D-Open','RESILIENT'):
        for fold in range(5):
            data={s:inputs(dataset,f,protocol) for s,f in [('validation',(fold+1)%5),('evaluation',fold)]}
            for seed in SEEDS:
                for name in METHODS:
                    resource_check('cpu');mark=time.perf_counter();key=name.replace(' ','_')
                    directory=BASE/dataset/f'fold{fold}'/f'seed{seed}'/key
                    result=OUT/dataset/f'fold{fold}'/f'seed{seed}'/key;record_path=result/'report.json'
                    if record_path.exists():registry.append(json.loads(record_path.read_text())['registry']);continue
                    trained=json.loads((directory/'report.json').read_text())
                    assert sha(directory/'estimator.joblib')==trained['estimator_sha256']
                    if name=='XGBoost':
                        recovered=recovery/dataset/f'fold{fold}'/f'seed{seed}'
                        repair=json.loads((recovered/'report.json').read_text())
                        assert repair['original_model_sha256']==trained['estimator_sha256']
                        assert sha(recovered/'estimator.ubj')==repair['model_sha256']
                        model=estimator(name,repair['settings'],seed);model.load_model(recovered/'estimator.ubj')
                    else:model=joblib.load(directory/'estimator.joblib')
                    transform=None
                    if name=='HYDRA':transform=torch.load(directory/'transform.pt',weights_only=False)
                    frames={};errors={}
                    saved=pd.read_csv(PRED/dataset/f'fold{fold}'/f'seed{seed}'/key/'predictions.csv',dtype={'person':str})
                    for split,(frame,raw,ids) in data.items():
                        if name=='HYDRA':
                            with torch.inference_mode():x=transform['scaler'].transform(transform['transformer'].batch(torch.from_numpy(raw),batch_size=16)).double().numpy()
                        else:x=tree_features(raw)
                        if name=='LightGBM':x=pd.DataFrame(x,columns=[f'f{i}' for i in range(x.shape[1])])
                        probabilities=probability(model,x,name)
                        full=frame.merge(ids.assign(probability=probabilities),on=['person','origin'],how='left',validate='one_to_one')
                        assert np.isfinite(full.probability).all()
                        known=full[full.label.notna()].merge(saved[saved.split==split],on=['person','origin'],suffixes=('','_saved'),validate='one_to_one')
                        assert len(known)==len(saved[saved.split==split])==full.label.notna().sum()
                        assert np.array_equal(known.label,known.label_saved)
                        errors[split]=float((known.probability-known.probability_saved).abs().max())
                        assert errors[split]<1e-6,(dataset,fold,seed,name,errors)
                        frames[split]=full
                    threshold=source_threshold(frames['validation'],'probability',cfg)
                    statistics,alerts,events=score_events(frames['evaluation'],'probability',threshold,cfg['silence_days'])
                    validation=score_events(frames['validation'],'probability',threshold,cfg['silence_days'])[0]
                    result.mkdir(parents=True,exist_ok=True)
                    pd.concat([v.assign(split=k) for k,v in frames.items()]).to_csv(result/'issue_days.csv',index=False)
                    alerts.to_csv(result/'alerts.csv',index=False);pd.DataFrame(events).to_csv(result/'events.csv',index=False)
                    record=dict(dataset=dataset,fold=fold,seed=seed,method=name,status='completed',seconds=time.perf_counter()-mark,**statistics)
                    save_json(record_path,dict(registry=record,validation=validation,known_prediction_max_error=errors,
                                              model_sha256=sha(directory/'estimator.joblib'),transform_sha256=sha(directory/'transform.pt') if name=='HYDRA' else None,manifest_sha256=sha(path)))
                    registry.append(record);pd.DataFrame(registry).to_csv(OUT/'run_registry.csv',index=False)
                    save_json(ART/'status.json',dict(status='running',completed_units=len(registry),expected_units=150,pid=os.getpid()))
                    print(json.dumps(dict(dataset=dataset,fold=fold,seed=seed,method=name,completed=len(registry),seconds=record['seconds'])),flush=True)
    save_json(ART/'status.json',dict(status='completed',completed_units=len(registry),expected_units=150,seconds=time.perf_counter()-start))


if __name__=='__main__':
    try:
        with threadpool_limits(limits=1):main()
    except BaseException as error:
        save_json(ART/'status.json',dict(status='failed',error=repr(error)));raise
