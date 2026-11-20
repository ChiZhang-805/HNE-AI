"""CPU-only descriptive analysis of existing, read-only development data."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from study_inputs import SOURCE

def longest(mask):
    best=run=0
    for x in mask:
        run=run+1 if x else 0;best=max(best,run)
    return best

def main():
    out=ROOT/'data';out.mkdir(exist_ok=True)
    people=[];summary={}
    for ds in ['RESILIENT','BrisT1D-Open']:
        base=SOURCE/'data/daily_study'/ds
        old=json.loads((base/'summary.json').read_text())
        for path in sorted(base.glob('*_daily.npz')):
            with np.load(path,allow_pickle=False) as a:
                x=a['states']; fam=a['state_families'];obs=np.isfinite(x)
                names=np.unique(fam)
                n_fam=np.stack([obs[:,fam==f].any(1) for f in names],axis=1).sum(1)
                valid=a['eligible_origin'];y=a['future_labels'];positive=np.flatnonzero(valid & (y==1))
                clusters=0 if len(positive)==0 else 1+int(np.sum(np.diff(positive)>7))
                row={'dataset':ds,'person_id':str(a['person_id']),'fold':int(a['fold']),
                     'calendar_days':len(x),'days_with_two_families':int((n_fam>=2).sum()),
                     'longest_two_family_run_days':longest(n_fam>=2),
                     'longest_all_state_run_days':longest(obs.all(1)),
                     'state_observed_fraction':float(obs.mean()),
                     'context_observed_fraction':float(np.isfinite(a['contexts']).mean()),
                     'eligible_origins':int(valid.sum()),'positives':int((valid & (y==1)).sum()),
                     'positive_window_clusters_gap7':clusters}
                people.append(row)
                state_names=a['state_names'].tolist();family_map=fam.tolist();context_names=a['context_names'].tolist()
        frame=pd.DataFrame([r for r in people if r['dataset']==ds])
        summary[ds]={'source_root':str(base),'source_summary':{k:old[k] for k in ['people','people_with_samples','samples','positive','negative','state_features','context_features','protocol_sha256']},
                     'state_names':state_names,'state_families':family_map,'context_names':context_names,
                     'calendar_days':frame.calendar_days.describe().to_dict(),
                     'longest_two_family_run_days':frame.longest_two_family_run_days.describe().to_dict(),
                     'longest_all_state_run_days':frame.longest_all_state_run_days.describe().to_dict(),
                     'positive_people':int((frame.positives>0).sum()),
                     'positive_window_clusters_gap7':int(frame.positive_window_clusters_gap7.sum()),
                     'sha256':{name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ['people.csv','samples.csv','summary.json']}}
    pd.DataFrame(people).to_csv(out/'participant_coverage.csv',index=False)
    (out/'local_data_profile.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    for ds,s in summary.items():print(ds,json.dumps({k:v for k,v in s.items() if k in ['source_summary','calendar_days','longest_two_family_run_days','longest_all_state_run_days','positive_people','positive_window_clusters_gap7']},ensure_ascii=False))

if __name__=='__main__':main()
