"""CPU coverage upper bounds for guidance; reads no future outcome values."""
from pathlib import Path
import hashlib, json, csv, argparse, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from study_inputs import SOURCE
parser=argparse.ArgumentParser(); parser.add_argument('--context-min-hours',type=int,default=1)
context_min=parser.parse_args().context_min_hours
suffix='_uniform8' if context_min==8 else ''
families={'RESILIENT':['heart','respiration','hrv','heart','sleep'],
          'BrisT1D-Open':['glucose','heart','activity','activity','activity']}
rows=[]; summary={}; hashes={}
for ds,fams in families.items():
    for path in sorted((SOURCE/'data/development'/ds).glob('*_hourly.npz')):
        person=path.name.removesuffix('_hourly.npz')
        daily=SOURCE/'data/daily_study'/ds/(person+'_daily.npz')
        with np.load(daily,allow_pickle=False) as a: origins=np.flatnonzero(a['eligible_origin'])
        if not len(origins): continue
        with np.load(path,allow_pickle=False) as a: x=a['values']
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        counts=np.isfinite(x[28*24:56*24]).reshape(42,16,6).sum(1)
        target=counts[21:]>=8; prior=counts[20:41]>=np.array([8]*5+[context_min])[None]
        nodes=target.sum(0)>=12
        sources=prior.sum(0)>=12
        edges=(target[:,:,None]&prior[:,None,:]).sum(0)>=12
        np.fill_diagonal(edges,False)
        any_node=any_edge=two_families=0
        for d in origins:
            end=(int(d)+1)*24
            counts=np.isfinite(x[end-28*24:end]).reshape(42,16,6).sum(1)
            target=counts[21:]>=8; prior=counts[20:41]>=np.array([8]*5+[context_min])[None]
            n=target[:,:5]&nodes[None,:5]
            e=target[:,:5,None]&prior[:,None,:]&edges[None,:5,:]
            any_node+=int(n.any()); any_edge+=int(e.any())
            two_families+=int(len({fams[j] for j in range(5) if n[:,j].any()})>=2)
        rows.append(dict(dataset=ds,person_id=person,eligible_origins=len(origins),
                         calibratable_state_channels=int(nodes[:5].sum()),
                         calibratable_state_families=len({fams[j] for j in range(5) if nodes[j]}),
                         calibratable_context_sources=int(sources[5:].sum()),
                         calibratable_context_edges=int(edges[:5,5:].sum()),
                         calibratable_directed_edges=int(edges[:5].sum()),
                         origins_any_node=any_node,origins_any_edge=any_edge,
                         origins_two_state_families=two_families))
    part=[r for r in rows if r['dataset']==ds]
    summary[ds]=dict(eligible_people=len(part),eligible_origins=sum(r['eligible_origins'] for r in part),
                    people_any_state_calibration=sum(r['calibratable_state_channels']>0 for r in part),
                    people_two_state_families=sum(r['calibratable_state_families']>=2 for r in part),
                    people_context_source_calibration=sum(r['calibratable_context_sources']>0 for r in part),
                    people_context_edge_calibration=sum(r['calibratable_context_edges']>0 for r in part),
                    origins_any_node=sum(r['origins_any_node'] for r in part),
                    origins_any_edge=sum(r['origins_any_edge'] for r in part),
                    origins_two_state_families=sum(r['origins_two_state_families'] for r in part))
report=dict(status='observed_coverage_only_not_model_quality',minimum_state_target_and_source_hours=8,minimum_context_source_hours=context_min,minimum_calibration_patches=12,
            calibration_patches_available=21,source_hourly_sha256=hashes,summary=summary,
            limitation='Upper bounds only: excludes constant-scale checks, fitted S quality, and future event labels. Uses existing origin eligibility metadata.')
(ROOT/('data/guidance_feasibility'+suffix+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
with (ROOT/('data/guidance_feasibility'+suffix+'.csv')).open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(summary))
