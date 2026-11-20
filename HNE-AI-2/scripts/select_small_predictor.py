"""Source-only forecast-loss search for S, before any event guidance refit.

Three fixed learning rates, identical source-person exclusions and initialization.
Selection uses only the already designated inner source holdout's forecast loss.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import psutil
import torch
from train_evidence import fit_small_components
from train_comparison import resource_check,sha,save_json
from training_data import load_people,component_assignments
BASE=ROOT/'artifacts/key_comparison/cpu_pilot/BrisT1D-Open/fold0/seed20261114'
ART=ROOT/'artifacts/teacher_selection'
RATES=(.0003,.001,.003)


def main():
    torch.set_num_threads(2)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu');started=time.perf_counter();ART.mkdir(parents=True,exist_ok=True)
    source=json.loads((BASE/'manifest.json').read_text());config={**source['config'],'small_epochs':80}
    people=load_people('BrisT1D-Open',0,only_split='train');assignments=component_assignments(people,20261114)
    assert all(assignments[p.person]==source['small_component'][p.person] for p in people['train'])
    provenance=dict(rates=RATES,max_epochs=80,patience=10,criterion='Existing source-inner heldout forecast loss; no event labels or target people',
                    data_sha256={p:sha(p) for group in people.values() for person in group for p in person.source_files},
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/train_evidence.py',ROOT/'src/precursor_guidance.py',ROOT/'src/training_data.py',ROOT/'src/study_inputs.py',BASE/'manifest.json']})
    provenance=json.loads(json.dumps(provenance));manifest=ART/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:raise RuntimeError('Teacher study provenance changed; preserve prior fits')
    save_json(manifest,provenance);save_json(ART/'status.json',dict(status='running',completed_candidates=0,pid=os.getpid()))
    all_reports=[]
    def log(event,**values):print(json.dumps(dict(event=event,**values)),flush=True)
    for candidate,rate in enumerate(RATES):
        resource_check('cpu');_,report=fit_small_components(people,assignments,{**config,'small_lr':rate},ART/f'candidate{candidate}',log)
        all_reports.append(report);save_json(ART/'status.json',dict(status='running',completed_candidates=candidate+1,pid=os.getpid()))
    selected=[];directory=ART/'selected';directory.mkdir(exist_ok=True)
    for component in range(3):
        candidates=[min(r[component]['curve'],key=lambda e:e['held_out_forecast_nll']) for r in all_reports]
        best=min(range(3),key=lambda c:candidates[c]['held_out_forecast_nll'])
        saved=torch.load(ART/f'candidate{best}'/f'component{component}.pt',weights_only=False)
        saved['selection']=dict(candidate=best,small_lr=RATES[best],manifest_sha256=sha(manifest))
        torch.save(saved,directory/f'component{component}.pt')
        selected.append(dict(component=component,candidate=best,small_lr=RATES[best],epochs=saved['report']['selected_epochs'],
                             source_holdout_forecast_loss=candidates[best]['held_out_forecast_nll'],
                             trial_losses=[x['held_out_forecast_nll'] for x in candidates],
                             checkpoint_sha256=sha(directory/f'component{component}.pt')))
    save_json(ART/'selection.json',dict(selected=selected,seconds=time.perf_counter()-started,
                                      scope='One source fold and seed, forecast-only teacher search; no event efficacy claim'))
    save_json(ART/'status.json',dict(status='completed',completed_candidates=3,components=9,seconds=time.perf_counter()-started))
    print(json.dumps(selected),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save_json(ART/'status.json',dict(status='failed',error=repr(error)));raise
