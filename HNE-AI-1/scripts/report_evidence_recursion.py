"""Two publication tables from paired, participant-held predictions."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analyze_hne_joint import dataset_sources
from analyze_hne_routed import fold_auc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/evidence_recursion'
CFG=json.loads((ROOT/'configs/hne_evidence_recursion.json').read_text())
KEY=['person_id','origin_day','fold','y']
R,B=CFG['datasets']
COLLEGE='College Experience'
HNE_LABEL = 'HNE (Ours)'
HNE_TEX = rf'\textbf{{{HNE_LABEL}}}'
SOURCES={'DNB daily':'DNB','FCM daily':'FCM','CATCH daily':'CATCH',
         'Chronos-2 FT':'Chronos-2 FT','moirai2':'Moirai 2.0 + LR','jape':'JAPE',
         'mantis':'Mantis + LR', 'timesfm':'TimesFM 2.5 + LR','feedback':HNE_LABEL}
RECENT = {'moirai2':'moirai2_warning', 'jape':'jape_development', 'mantis':'mantis_warning',
          'timesfm':'timesfm_warning'}


def read_csv(path):
    return pd.read_csv(path,dtype={'person_id':str})


def comparison_scores(dataset):
    # Retain the established common-origin cohort, including both DNB eligibility masks.
    sources = dataset_sources(dataset)
    common = None
    for name in ('DNB daily', 'FCM daily', 'CATCH daily', 'Chronos-2 FT', 'DNB 3h', 'DNB-tuned daily'):
        folder, _ = sources[name]
        runs = [read_csv(folder/f'fold{k}/held_person.csv') for k in range(5)]
        frame = pd.concat(runs, ignore_index=True)
        score = 'probability' if 'probability' in frame else 'score'
        frame = frame[KEY+[score]].dropna(subset=[score]).rename(columns={score:name})
        common = frame if common is None else common.merge(frame, on=KEY, validate='one_to_one')
    return common


def ensemble(source, variant, target=False, anchored=True):
    folds=[]
    base=OUT/'anchored' if anchored else OUT
    for fold in range(5):
        name='transfer' if target else 'held'
        runs=[read_csv(base/source/variant/f'fold{fold}'/f'seed{s}'/f'{name}.csv').set_index(KEY).sort_index() for s in CFG['seeds']]
        if not all(frame.index.is_unique and frame.index.equals(runs[0].index) for frame in runs):
            raise ValueError('Origin or label mismatch between seeds')
        avg=(sum(runs)/len(runs)).reset_index()
        # Match inference budget: three source seeds; source fold k predicts destination fold k.
        folds.append(avg[avg.fold==fold])
    return pd.concat(folds,ignore_index=True)


def intervals(frame, methods):
    # Positive/negative rank credits by person pair avoid an origins-by-origins matrix.
    people=sorted(frame.person_id.unique())
    index={p:i for i,p in enumerate(people)}
    credits=np.zeros((len(methods),len(people),len(people)))
    pairs=np.zeros(credits.shape[1:])
    for _,g in frame.groupby('fold'):
        positives=[(index[p],v[methods].to_numpy().T) for p,v in g[g.y==1].groupby('person_id')]
        negatives=[(index[p],np.sort(v[methods].to_numpy().T,axis=1)) for p,v in g[g.y==0].groupby('person_id')]
        for i,positive in positives:
            for j,negative in negatives:
                pairs[i,j]+=positive.shape[1]*negative.shape[1]
                for k,(a,b) in enumerate(zip(positive,negative)):
                    credits[k,i,j]+=(np.searchsorted(b,a,side='left').sum()+np.searchsorted(b,a,side='right').sum())/2
    rng=np.random.default_rng(20261113)
    n=len(people)
    weights=rng.multinomial(n,np.full(n,1/n),size=2000)
    denom=np.einsum('bi,ij,bj->b',weights,pairs,weights,optimize=True)
    numer=np.einsum('bi,mij,bj->bm',weights,credits,weights,optimize=True)
    return pd.DataFrame(numer[denom>0]/denom[denom>0,None],columns=methods)


def paired(left,right):
    joined=left.merge(right,on=KEY,validate='one_to_one')
    if len(joined)!=len(left):
        raise ValueError('Expected complete overlap on reference forecast origins')
    return joined


def metrics():
    main=[]; contrasts=[]; recent_contrasts=[]
    def contrast(frame,boot,dataset,label,a,b=None):
        values=boot[a] if b is None else boot[a]-boot[b]
        delta=fold_auc(frame,a)-(fold_auc(frame,b) if b else 0.)
        low,high=np.quantile(values,[.025,.975])
        contrasts.append(dict(dataset=dataset,comparison=label,value=delta,low=low,high=high))
    for dataset,other in [(R,B),(B,R)]:
        frame=ensemble(dataset,'feedback').rename(columns={'risk':'feedback'})
        for v in ('one_step','fixed_query','reset'):
            frame=paired(frame,ensemble(dataset,v).rename(columns={'risk':v}))
        for v in ('feedback','fixed_query'):
            frame=paired(frame,ensemble(other,v,True)[KEY+['risk']].rename(columns={'risk':'source_'+v}))
            plain=ensemble(dataset,v,anchored=False)[KEY+['risk']]
            frame=paired(frame,plain.rename(columns={'risk':'unanchored' if v=='feedback' else 'unanchored_fixed'}))
        foreign='source_feedback'
        cols=['feedback','one_step','fixed_query','reset','omit_H','omit_N','omit_E',foreign,'source_fixed_query','unanchored','unanchored_fixed','anchor']
        draws=intervals(frame,cols)
        for col in ['omit_H','omit_N','omit_E','fixed_query','one_step','reset']:
            contrast(frame,draws,dataset,'full - '+col,'feedback',col)
        contrast(frame,draws,dataset,'full - anchor','feedback','anchor')
        contrast(frame,draws,dataset,'direct - direct fixed','unanchored','unanchored_fixed')
        contrast(frame,draws,dataset,'cross AUROC',foreign)
        contrast(frame,draws,dataset,'cross - fixed query',foreign,'source_fixed_query')
        frame.to_csv(OUT/f'{dataset}_mechanism.csv.gz',index=False)
        comparison=comparison_scores(dataset)
        common=paired(comparison,frame[KEY+['feedback','anchor','unanchored']])
        for method, folder in RECENT.items():
            predictions=read_csv(ROOT/'results'/folder/dataset/'held_person.csv')
            common=paired(common,predictions[KEY+['score']].rename(columns={'score':method}))
        new_boot=intervals(common,['feedback',*RECENT])
        for method in RECENT:
            low,high=np.quantile(new_boot.feedback-new_boot[method],[.025,.975])
            recent_contrasts.append(dict(dataset=dataset,method=SOURCES[method],
                hne_minus_method=fold_auc(common,'feedback')-fold_auc(common,method),low=low,high=high))
        common.to_csv(OUT/f'{dataset}_comparison.csv.gz',index=False)
        for k,label in SOURCES.items():
            ap=sum(len(g)*average_precision_score(g.y,g[k]) for _,g in common.groupby('fold'))/len(common)
            main.append(dict(dataset=dataset,method=label,auroc=fold_auc(common,k),ap=ap))
    ext=read_csv(OUT/'anchored/college_transfer.csv.gz')
    for source in [R,B]:
        for mode in ['feedback','fixed_query']:
            ext[mode]=ext[[f'{source}/{mode}/fold{i}' for i in range(5)]].mean(axis=1)
        draws=intervals(ext,['feedback','fixed_query'])
        contrast(ext,draws,COLLEGE,source+' to college','feedback')
        contrast(ext,draws,COLLEGE,source+' gain to college','feedback','fixed_query')
    first=pd.DataFrame(main);second=pd.DataFrame(contrasts)
    first.to_csv(OUT/'table1.csv',index=False);second.to_csv(OUT/'table2.csv',index=False)
    pd.DataFrame(recent_contrasts).to_csv(OUT/'recent_baselines_paired.csv',index=False)


MAIN_ROWS = [
    ('DNB', r'DNB \href{https://doi.org/10.1038/srep00342}{(Chen et al., 2012)}'),
    ('FCM', r'FCM \href{https://arxiv.org/abs/2410.12206}{(Zhao et al., 2024)}'),
    ('CATCH', r'CATCH \href{https://proceedings.iclr.cc/paper_files/paper/2025/file/2b25c39788e5cf11d3541de433ebf4c0-Paper-Conference.pdf}{(Wu et al., 2025)}'),
    ('Chronos-2 FT', r'Chronos-2 FT \href{https://arxiv.org/abs/2510.15821}{(Ansari et al., 2025)}'),
    ('Moirai 2.0 + LR', r'Moirai 2.0 + LR \href{https://arxiv.org/abs/2511.11698}{(Liu et al., 2025)}'),
    ('JAPE', r'JAPE \href{https://arxiv.org/abs/2608.11801}{(Wei et al., 2026)}'),
    ('Mantis + LR', r'Mantis + LR \href{https://proceedings.mlr.press/v306/feofanov26a.html}{(Feofanov et al., 2026)}'),
    ('TimesFM 2.5 + LR', r'TimesFM 2.5 + LR \href{https://github.com/google-research/timesfm}{(Google Research, 2025)}'),
    (HNE_LABEL, HNE_TEX),
]

ABLATION_ROWS = [
    ('feedback', HNE_TEX),
    ('omit_H', 'w/o History'),
    ('omit_N', 'w/o Network'),
    ('omit_E', 'w/o Environment'),
    ('fixed_query', 'w/o Adaptive Query'),
    ('one_step', 'w/o Iteration'),
    ('anchor', 'w/o Latent Correction'),
]


def ablation_scores():
    rows = []
    for dataset in (R, B):
        common = read_csv(OUT/f'{dataset}_comparison.csv.gz')
        mechanism = read_csv(OUT/f'{dataset}_mechanism.csv.gz')
        frame = paired(common[KEY], mechanism)
        for column, _ in ABLATION_ROWS:
            mean_ap = sum(len(g) * average_precision_score(g.y, g[column])
                          for _, g in frame.groupby('fold')) / len(frame)
            rows.append((dataset, column, fold_auc(frame, column), mean_ap))
    scores = pd.DataFrame(rows, columns=['dataset', 'method', 'auroc', 'ap'])
    scores.to_csv(OUT/'ablations.csv', index=False)
    return scores


def tex_rows(scores, methods):
    indexed = scores.set_index(['dataset', 'method'])
    values = np.array([[indexed.loc[(dataset, key), metric]
                        for dataset in (R, B) for metric in ('auroc', 'ap')]
                       for key, _ in methods])
    displayed = np.round(values, 4)
    best = displayed.max(axis=0)
    lines = []
    for (_, label), row in zip(methods, displayed):
        cells = [f'{number:.4f}' if number < maximum else rf'\textbf{{{number:.4f}}}'
                 for number, maximum in zip(row, best)]
        lines.append(' & '.join([label, *cells]) + r' \\')
    return '\n'.join(lines)


def render():
    layout = (ROOT/'paper/tables/evidence_recursion_layout.tex').read_text(encoding='utf-8')
    layout = layout.replace('% MAIN_ROWS', tex_rows(read_csv(OUT/'table1.csv'), MAIN_ROWS))
    layout = layout.replace('% ABLATION_ROWS', tex_rows(read_csv(OUT/'ablations.csv'), ABLATION_ROWS))
    compiler = os.getenv('TECTONIC') or shutil.which('tectonic') or 'tectonic'
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix='layout-', dir=OUT) as temp:
        work = Path(temp)
        source = work/'HNE_AI_1_tables.tex'
        source.write_text(layout, encoding='utf-8')
        subprocess.run([compiler, str(source)], cwd=work, env=env, check=True)
        os.replace(source.with_suffix('.pdf'), OUT/'HNE_AI_1_tables.pdf')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout-only', action='store_true',
                        help='Render the two tables from the published aggregate CSVs.')
    if not parser.parse_args().layout_only:
        metrics()
        ablation_scores()
    render()
