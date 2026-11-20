"""Run all-person CPU baselines on fixed development folds, with source-only selection.

Preserves the prior sparse pilot. Six prespecified candidates per method, equal
person training weights, and complete evaluation origins. No APIs or GPU needed.
HYDRA uses its author's experimental multivariate transform and a ridge head;
sigmoid(margin) is an explicit probability link, not a claim of calibration.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'cache/public_baseline_packages'), str(ROOT/'literature/code_audit/HYDRA')]
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, recall_score
from threadpoolctl import threadpool_limits
from hydra import SparseScaler
from hydra_multivariate import HydraMultivariate
from run_public_baselines import read_windows, tree_features
from training_data import load_people
from train_comparison import metrics, save_json, sha, resource_check

OUT = ROOT/'results/local_baselines'
ART = ROOT/'artifacts/local_baselines'
METHODS = ('M01_HGB', 'Random Forest', 'XGBoost', 'LightGBM', 'HYDRA')
SEEDS = (20261113, 20261114, 20261115)


def candidate_grid(name):
    if name == 'Random Forest':
        return [dict(max_depth=d, min_samples_leaf=l) for d in (None, 4, 8) for l in (2, 5)]
    if name == 'XGBoost':
        return [dict(max_depth=d, learning_rate=l) for d in (2, 3, 5) for l in (.03, .1)]
    if name == 'LightGBM':
        return [dict(num_leaves=d, min_child_samples=l) for d in (7, 15, 31) for l in (5, 15)]
    if name == 'M01_HGB':
        return [dict(max_leaf_nodes=d, learning_rate=l) for d in (7, 15, 31) for l in (.03, .1)]
    return [dict(alpha=a) for a in (.01, .1, 1., 10., 100., 1000.)]


def estimator(name, settings, seed):
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed, **settings)
    if name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, n_jobs=1, device='cpu', tree_method='exact', random_state=seed, **settings)
    if name == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, learning_rate=.05, n_jobs=1, verbosity=-1, force_col_wise=True, random_state=seed, **settings)
    if name == 'M01_HGB':
        return HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=10, early_stopping=False, random_state=seed, **settings)
    return RidgeClassifier(solver='lsqr', tol=1e-4, max_iter=1000, **settings)


def probability(model, x, name):
    return expit(model.decision_function(x)) if name == 'HYDRA' else model.predict_proba(x)[:,1]


def score(ids, probabilities):
    records = ids.assign(probability=probabilities).to_dict('records')
    return {**metrics(records), 'accuracy':float(accuracy_score(ids.label, probabilities>=.5)),
            'sensitivity':float(recall_score(ids.label, probabilities>=.5, zero_division=0))}


def main():
    started=time.perf_counter()
    torch.set_num_threads(1)
    if os.name=='nt': psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    OUT.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'configs/daily_protocol.json').read_text())
    provenance=dict(protocol_sha256=sha(ROOT/'configs/daily_protocol.json'),
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'scripts/run_public_baselines.py',ROOT/'src/training_data.py',ROOT/'src/study_inputs.py',ROOT/'literature/code_audit/HYDRA/hydra.py',ROOT/'literature/code_audit/HYDRA/hydra_multivariate.py']},
                    seeds=SEEDS, folds=list(range(5)), datasets=['BrisT1D-Open','RESILIENT'],
                    grid={m:candidate_grid(m) for m in METHODS},
                    selection='validation_NLL; six fixed candidates; no evaluation-based tuning',
                    training='all legal training origins; equal person sample weights',
                    hydra_link='sigmoid ridge margin, not post-hoc calibrated',
                    exposure='Previously exposed development cohorts; not prospective untouched tests', threads=1, device='cpu')
    # JSON normalization keeps tuple/list representation identical on resume.
    provenance=json.loads(json.dumps(provenance))
    manifest=ART/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:
        raise RuntimeError('Local baseline provenance changed; existing work is preserved')
    save_json(manifest,provenance)
    registry=[]
    try:
        for dataset in provenance['datasets']:
            for fold in range(5):
                resource_check('cpu')
                people=load_people(dataset,fold,include_evaluation=True)
                inputs={}; ids={}
                for split in ('train','validation','evaluation'):
                    inputs[split],ids[split]=read_windows(people[split],protocol,maximum=0)
                assert all(not (set(ids[a].person)&set(ids[b].person)) for a,b in [('train','validation'),('train','evaluation'),('validation','evaluation')])
                tree={s:tree_features(x) for s,x in inputs.items()}
                counts=ids['train'].person.value_counts()
                weights=ids['train'].person.map(lambda p:len(ids['train'])/(len(counts)*counts[p])).to_numpy()
                if ids['train'].label.nunique()!=2:
                    raise RuntimeError('Single-class training split requires a recorded prespecified fallback')
                data_hashes={p:sha(p) for persons in people.values() for person in persons for p in person.source_files}
                for seed in SEEDS:
                    transformed=None; transformer=scaler=None
                    for name in METHODS:
                        key=name.replace(' ','_')
                        run=ART/dataset/f'fold{fold}'/f'seed{seed}'/key
                        result=OUT/dataset/f'fold{fold}'/f'seed{seed}'/key
                        done=run/'report.json'
                        if done.exists():
                            report=json.loads(done.read_text())
                            if report['data_sha256']!=data_hashes or report['manifest_sha256']!=sha(manifest):
                                raise RuntimeError('Completed baseline inputs changed; existing work is preserved')
                            registry.append(report['registry']); continue
                        resource_check('cpu'); mark=time.perf_counter()
                        x=tree
                        if name=='HYDRA':
                            torch.manual_seed(seed)
                            transformer=HydraMultivariate(inputs['train'].shape[-1],inputs['train'].shape[1])
                            scaler=SparseScaler()
                            with torch.inference_mode():
                                transformed={'train':scaler.fit_transform(transformer.batch(torch.from_numpy(inputs['train']),batch_size=16)).double().numpy()}
                                for s in ('validation','evaluation'):
                                    transformed[s]=scaler.transform(transformer.batch(torch.from_numpy(inputs[s]),batch_size=16)).double().numpy()
                            x=transformed
                        if name=='LightGBM':
                            columns=[f'f{i}' for i in range(x['train'].shape[1])]
                            x={s:pd.DataFrame(v,columns=columns) for s,v in x.items()}
                        trials=[]; best=float('inf'); selected=None; best_model=None
                        for index, settings in enumerate(candidate_grid(name)):
                            trial=time.perf_counter(); model=estimator(name,settings,seed)
                            model.fit(x['train'],ids['train'].label.to_numpy(),sample_weight=weights)
                            val=score(ids['validation'],probability(model,x['validation'],name))
                            trials.append(dict(candidate=index,settings=settings,validation=val,seconds=time.perf_counter()-trial))
                            if val['nll']<best:
                                best,best_model,selected=val['nll'],model,index
                        result.mkdir(parents=True,exist_ok=True); run.mkdir(parents=True,exist_ok=True)
                        frames=[]; statistics={}
                        for split in ('validation','evaluation'):
                            p=probability(best_model,x[split],name)
                            frames.append(ids[split].assign(split=split,fold=fold,seed=seed,method=name,probability=p,predicted=(p>=.5).astype(int)))
                            statistics[split]=score(ids[split],p)
                        pd.concat(frames).to_csv(result/'predictions.csv',index=False)
                        ids['train'].to_csv(run/'training_origins.csv',index=False)
                        joblib.dump(best_model,run/'estimator.joblib')
                        if name=='HYDRA': torch.save(dict(transformer=transformer,scaler=scaler),run/'transform.pt')
                        record=dict(dataset=dataset,fold=fold,seed=seed,method=name,status='completed',
                                    train_origins=len(ids['train']),validation_origins=len(ids['validation']),evaluation_origins=len(ids['evaluation']),
                                    candidates=len(trials),selected_candidate=selected,seconds=time.perf_counter()-mark,
                                    auroc=statistics['evaluation']['auroc'],auprc=statistics['evaluation']['auprc'],nll=statistics['evaluation']['nll'])
                        report=dict(registry=record,trials=trials,metrics=statistics,data_sha256=data_hashes,
                                    manifest_sha256=sha(manifest),estimator_sha256=sha(run/'estimator.joblib'))
                        save_json(done,report); registry.append(record)
                        pd.DataFrame(registry).to_csv(OUT/'run_registry.csv',index=False)
                        save_json(ART/'status.json',dict(status='running',pid=os.getpid(),completed_units=len(registry),expected_units=150))
                        print(json.dumps(record),flush=True)
        save_json(ART/'status.json',dict(status='completed',completed_units=len(registry),expected_units=150,
                                        invocation_seconds=time.perf_counter()-started,device='cpu',threads=1))
        print(json.dumps(dict(event='completed',units=len(registry),seconds=time.perf_counter()-started)),flush=True)
    except BaseException as error:
        save_json(ART/'status.json',dict(status='failed',error=repr(error),completed_units=len(registry)))
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
