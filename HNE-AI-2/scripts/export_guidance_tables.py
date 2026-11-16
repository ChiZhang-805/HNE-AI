"""Export standalone English LaTeX tables and their PDFs from the saved study results."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'tables'
TECTONIC = shutil.which('tectonic')
REPORTED = ROOT / 'results/reported'

TRANSFER = ROOT / 'results' / 'guidance_transfer'

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=216mm,paperheight=HEIGHTmm,left=10mm,right=10mm,top=6mm,bottom=4mm]{geometry}
\usepackage{booktabs,multirow}
\ifdefined\XeTeXversion
  \DeclareFontShape{TU}{lmr}{bx}{n}{<-> \UnicodeFontFile{lmroman10-bold}{\UnicodeFontTeXLigatures}}{}
\fi
\pagestyle{empty}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.19}
\setlength{\heavyrulewidth}{0.9pt}
\setlength{\lightrulewidth}{0.45pt}
\setlength{\cmidrulewidth}{0.35pt}
\makeatletter
\long\def\@makecaption#1#2{{\centering\normalsize\bfseries #1.\enspace #2\par}\vspace{6pt}}
\makeatother
\newcommand{\unmeasured}{\mbox{-}\kern.06em\mbox{-}}
\begin{document}
"""


def table(number, title, label, groups, columns, row_heading, rows, height):
    n = len(columns)
    lines = [
        PREAMBLE.replace('HEIGHT', str(height)).rstrip(),
        rf'\setcounter{{table}}{{{number - 1}}}',
        r'\begin{table}[!ht]',
        r'\centering\small',
        rf'\caption{{{title}}}',
        rf'\label{{tab:{label}}}',
        rf'\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}l*{{{2*n}}}{{c}}@{{}}}}',
        r'\toprule',
        ' & '.join([rf'\multirow{{2}}{{*}}{{{row_heading}}}'] +
                   [rf'\multicolumn{{{n}}}{{c}}{{{g}}}' for g in groups]) + r' \\',
        rf'\cmidrule(lr){{2-{n+1}}}\cmidrule(l){{{n+2}-{2*n+1}}}',
        ' & ' + ' & '.join(columns * 2) + r' \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(' & '.join(row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}', r'\end{table}', r'\end{document}'])
    return '\n'.join(lines) + '\n'


def main_results(datasets, scores):
    refs = {
        'Random Forest': ('breiman2001', 'Breiman, 2001', 'https://doi.org/10.1023/A:1010933404324'),
        'XGBoost': ('chen2016', r'Chen \& Guestrin, 2016', 'https://doi.org/10.1145/2939672.2939785'),
        'LightGBM': ('ke2017', 'Ke et al., 2017', 'https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree'),
        'HYDRA': ('dempster2023', 'Dempster et al., 2023', 'https://doi.org/10.1007/s10618-023-00939-3'),
    }
    source, target = datasets
    methods = sorted(refs, key=lambda m: (-scores[source][m]['accuracy'], -scores[source][m]['auroc'], m))
    rows = []
    for method in [*methods, 'SAGE-TS']:
        if method in refs:
            key, authors, url = refs[method]
            name = 'HYDRA (multivariate)' if method == 'HYDRA' else method
            label = rf'{name}~\methodcite{{{key}}}{{{authors}}}{{{url}}}'
        else:
            label = r'\textbf{SAGE-TS (Ours)}'
        cells = [label]
        for dataset in datasets:
            m = scores[dataset][method]
            cells.extend([f"{100*m['accuracy']:.2f}", f"{100*m['sensitivity']:.2f}",
                          f"{m['auroc']:.4f}", f"{m['auprc']:.4f}"])
        rows.append(cells)
    columns = [r'Acc. (\%)', r'Sens. (\%)', 'AUROC', 'AUPRC']
    output = table(1, 'Early warning of sustained abnormalities on a single validation fold',
                   'guidance-results', [source, target + ' (frozen transfer)'], columns, 'Method', rows, height=56)
    # Standalone citations link to the original publications. When the float is
    # copied to a paper, its fallback uses the native citation command and BibTeX.
    return output.replace(r'\begin{document}', r'''\usepackage[hidelinks]{hyperref}
\newcommand{\methodcite}[3]{\href{#3}{[#2]}}
\begin{document}''').replace(r'\centering\small', r'''\providecommand{\methodcite}[3]{\cite{#1}}
\centering\small''')


def within_person_results(datasets):
    # Pair only dates from the same participant. A single-class participant
    # remains in the per-person export and contributes zero pairs to the AUROC.
    records = []
    for dataset in datasets:
        predictions = pd.read_csv(TRANSFER / (dataset + '_predictions.csv'), dtype={'person': str})
        for person, group in predictions.groupby('person'):
            positives = int(group.label.sum())
            pairs = positives * (len(group) - positives)
            for method in ('baseline', 'aligned', 'reversed', 'full_fusion'):
                auc = roc_auc_score(group.label, group[method]) if pairs else None
                records.append(dict(dataset=dataset, person=person, method=method, origins=len(group),
                                    positives=positives, pairs=pairs, auroc=auc))
    return pd.DataFrame(records)


def ablations(report, datasets, within):
    variants = [('Aligned guidance', 'aligned'), ('Guidance removed', 'baseline'),
                ('Evidence order reversed', 'reversed')]
    rows = []
    for name, variant in variants:
        cells = [name]
        for dataset in datasets:
            value = within[dataset][variant]['auroc']
            auc = f'{value:.4f}' if value is not None else r'\unmeasured'
            if variant == 'aligned':
                cells.extend([auc, '0.000', '0.000'])
            else:
                # Saved contrasts are aligned minus comparator. The ablation table
                # reports the intervention minus the aligned reference, in 10^-5 units.
                contrast = report[dataset]['contrasts'][variant]
                nll_change = -contrast['delta_nll'] * 1e5
                prob_change = contrast['mean_absolute_probability_change'] * 1e5
                cells.extend([auc, f'${nll_change:+.3f}$', f'{prob_change:.3f}'])
        rows.append(cells)
    columns = [r'\shortstack{Within-person\\AUROC $\uparrow$}',
               r'\shortstack{$\Delta$NLL\\($10^{-5}$)}',
               r'\shortstack{Mean $|\Delta p|$\\($10^{-5}$)}']
    source, target = datasets
    return table(2, 'Inference ablations relative to aligned guidance', 'guidance-ablations',
                 [source, target + ' (frozen transfer)'], columns, 'Variant', rows, height=50)


def export(from_run=False):
    if not TECTONIC:
        raise FileNotFoundError('The table exporter requires Tectonic.')
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    cfg = json.loads((ROOT / 'configs/guidance_transfer.json').read_text(encoding='utf8'))
    datasets = (cfg['source_dataset'], cfg['target_dataset'])
    if from_run:
        report_file = TRANSFER / 'summary.json'
        baseline_file = ROOT / 'results/public_baselines/summary.json'
        within_rows = within_person_results(datasets)
        within = {d: {} for d in datasets}
        for (dataset, method), group in within_rows.groupby(['dataset', 'method']):
            pairs = group.pairs.sum()
            within[dataset][method] = {'auroc': (group.auroc * group.pairs).sum() / pairs if pairs else None}
    else:
        report_file = REPORTED / 'frozen_transfer.json'
        baseline_file = REPORTED / 'public_baselines.json'
        within = json.loads((REPORTED / 'within_person.json').read_text(encoding='utf8'))
    report = json.loads(report_file.read_text(encoding='utf8'))
    scores = json.loads(baseline_file.read_text(encoding='utf8'))['metrics']
    tables = {
        'guidance_results': main_results(datasets, scores),
        'guidance_ablations': ablations(report, datasets, within),
    }
    with tempfile.TemporaryDirectory(dir=cache) as working_dir:
        work = Path(working_dir)
        for name, table in tables.items():
            tex = work / (name + '.tex')
            tex.write_text(table, encoding='ascii')
            subprocess.run([str(TECTONIC), '--keep-logs', tex.name], cwd=work, env=env, check=True)
            log = tex.with_suffix('.log').read_text(encoding='utf8', errors='replace')
            problem = re.search(r'^.*(?:Overfull \\[hv]box|Missing character:|Float too large for page).*$', log, re.M)
            if problem:
                raise ValueError('Check the exported table: ' + problem.group())
        if from_run:
            within_rows.to_csv(work / 'within_person.csv', index=False)
            os.replace(work / 'within_person.csv', TRANSFER / 'within_person.csv')
        for name in tables:
            for extension in ('.tex', '.pdf'):
                os.replace(work / (name + extension), OUT / (name + extension))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', action='store_true', help='Use privately regenerated predictions and metrics.')
    export(parser.parse_args().from_run)
