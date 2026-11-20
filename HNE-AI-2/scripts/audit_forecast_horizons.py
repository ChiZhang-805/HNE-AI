"""Measure horizon feasibility without modifying labels or training models."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from protocol import forecast_origins
from study_inputs import SOURCE


def main():
    cfg=json.loads((ROOT/'configs/daily_protocol.json').read_text());rows=[];origins=[];hashes={};checks=0
    for dataset in ('BrisT1D-Open','RESILIENT'):
        for path in sorted((SOURCE/'data/daily_study'/dataset).glob('*_daily.npz')):
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as a:
                blocks={name:a['block_'+name] for name in ('label','known_variables','known_families','abnormal_families','unknown_families')}
                results={}
                for gap in (1,3,7,14):
                    spec={**cfg,'forecast_gap_days':gap,'future_start_offset':gap+1,'future_end_offset':gap+7}
                    result=forecast_origins(blocks,a['available_at'],a['dates'],spec);results[gap]=result
                    if gap==1:
                        assert np.array_equal(result['eligible_origin'],a['eligible_origin'])
                        assert np.allclose(result['future_labels'],a['future_labels'],equal_nan=True)
                        checks+=1
                intersection=np.logical_and.reduce([r['eligible_origin'] for r in results.values()])
                for gap,r in results.items():
                    legal=r['eligible_origin'];issue=(blocks['label']==0)&~np.isnat(r['issue_time'])
                    known=np.isfinite(r['future_labels']);positive=legal&(r['future_labels']==1)
                    rows.append(dict(dataset=dataset,person=str(a['person_id']),fold=int(a['fold']),gap_days=gap,
                                     eligible_reference_families=int(a['reference_eligible_families'].sum()),
                                     issue_origins=int(issue.sum()),scorable_origins=int(legal.sum()),unknown_outcomes=int((issue&~known).sum()),
                                     positives=int(positive.sum()),intersection_origins=int(intersection.sum()),
                                     intersection_positives=int((intersection&(r['future_labels']==1)).sum())))
                    for d in np.flatnonzero(issue):
                        origins.append(dict(dataset=dataset,person=str(a['person_id']),fold=int(a['fold']),gap_days=gap,origin=int(d),
                                            label=float(r['future_labels'][d]) if known[d] else '',intersection=bool(intersection[d])))
    out=ROOT/'results/horizon_audit';out.mkdir(parents=True,exist_ok=True)
    for name,data in [('participants',rows),('origins',origins)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    aggregate=[]
    for dataset in ('BrisT1D-Open','RESILIENT'):
        for gap in (1,3,7,14):
            part=[r for r in rows if r['dataset']==dataset and r['gap_days']==gap]
            aggregate.append(dict(dataset=dataset,gap_days=gap,people=sum(r['scorable_origins']>0 for r in part),
                                  positive_people=sum(r['positives']>0 for r in part),
                                  **{k:sum(r[k] for r in part) for k in ('issue_origins','scorable_origins','unknown_outcomes','positives','intersection_origins','intersection_positives')}))
    report=dict(scope='Feasibility and label sensitivity only; no new prediction performance; fixed seven-day multi-family endpoint',
                gap_one_exact_matches=checks,protocol_sha256=hashlib.sha256((ROOT/'configs/daily_protocol.json').read_bytes()).hexdigest(),
                source_sha256=hashes,summary=aggregate)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
