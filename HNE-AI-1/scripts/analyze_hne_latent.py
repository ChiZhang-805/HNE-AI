"""Align predictions, summarize continuous-feedback experiments, and draw results."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_hne_joint import dataset_sources, KEY
from analyze_hne_routed import fold_auc, clustered_auc_bootstrap

LABELS = {'recurrent': 'HNE latent', 'single_pass': 'Single pass', 'reset_feedback': 'Reset feedback',
          'detached_feedback': 'Detached feedback', 'previous_HNE': 'Additive HNE',
          'frozen_core': 'Frozen core', 'no_history_coordinates': 'Without H coordinates',
          'no_network_coordinates': 'Without N coordinates', 'no_environment_coordinates': 'Without E/NE coordinates',
          'no_personal_reference': 'Without reference token'}


def load(folder, folds, column=None):
    frames = []
    for k in folds:
        path = folder/f'fold{k}/held_person.csv'
        if not path.exists(): return None
        f = pd.read_csv(path, dtype={'person_id': str})
        f['value'] = f[column or ('probability' if 'probability' in f else 'score')]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def auxiliary(root, output, config):
    state_rows = []; state_differences = []; input_rows = []; transfer_rows = []
    for dataset in config['datasets']:
        frames = []
        for k in config['folds']:
            files = sorted((root/dataset/'recurrent'/f'fold{k}').glob('seed*/state_interventions/predictions.csv'))
            if not files: continue
            values = [pd.read_csv(p, dtype={'person_id': str}) for p in files]
            frame = values[0].copy()
            for col in ['depth1', 'depth2', 'depth4', 'depth8', 'reset', 'shuffle']:
                frame[col] = np.mean([v[col].to_numpy() for v in values], axis=0)
            frames.append(frame)
        if frames:
            paired = pd.concat(frames, ignore_index=True)
            methods = ['depth1', 'depth2', 'depth4', 'depth8', 'reset', 'shuffle']
            boot = clustered_auc_bootstrap(paired, methods, 2000)
            for j, method in enumerate(methods):
                low, high = np.nanquantile(boot[:, j], [.025, .975])
                state_rows.append(dict(dataset=dataset, method=method, folds=len(frames), n=len(paired),
                                       auroc=fold_auc(paired, method), low=low, high=high))
                low, high = np.nanquantile(boot[:, 2]-boot[:, j], [.025, .975])
                state_differences.append(dict(dataset=dataset, method=method,
                     delta=fold_auc(paired, 'depth4')-fold_auc(paired, method), low=low, high=high))
            paired.to_csv(output/f'{dataset}_state_predictions.csv', index=False)
        for condition in ['within_day_shuffle', 'day_order_shuffle', 'channel_day_shift']:
            for variant in config['variants']:
                files = sorted((root/'input_interventions'/dataset/condition/variant).glob('fold*/held_person.csv'))
                if not files: continue
                paired = pd.concat([pd.read_csv(p, dtype={'person_id': str}) for p in files], ignore_index=True)
                folds = sorted(paired.fold.unique()); original = load(root/dataset/variant, folds)
                paired = paired.merge(original[KEY+['value']], on=KEY, validate='one_to_one')
                boot = clustered_auc_bootstrap(paired, ['value', 'probability'], 2000)
                low, high = np.nanquantile(boot[:, 1]-boot[:, 0], [.025, .975])
                input_rows.append(dict(dataset=dataset, condition=condition, variant=variant, folds=len(files),
                    original_auroc=fold_auc(paired, 'value'), intervened_auroc=fold_auc(paired, 'probability'),
                    delta=fold_auc(paired, 'probability')-fold_auc(paired, 'value'), low=low, high=high))
    for path in (root/'transfer').glob('*/*/metrics.json'):
        info = json.loads(path.read_text()); transfer_rows.append(info)
    pd.DataFrame(state_rows).to_csv(output/'rq2_state_effects.csv', index=False)
    pd.DataFrame(state_differences).to_csv(output/'rq2_state_differences.csv', index=False)
    pd.DataFrame(input_rows).to_csv(output/'rq1_input_effects.csv', index=False)
    pd.DataFrame(transfer_rows).to_csv(output/'rq3_transfer.csv', index=False)
    for name, rows, key, values in [('rq2_states', state_rows, 'method', ['depth1', 'depth2', 'depth4', 'depth8', 'reset', 'shuffle'])]:
        if not rows: continue
        table = pd.DataFrame(rows); fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
        for ax, ds in zip(axes, config['datasets']):
            part = table[table.dataset.eq(ds)].set_index(key)
            names = [n for n in values if n in part.index]
            ax.bar(np.arange(len(names)), part.loc[names, 'auroc'], color=['black' if n=='depth4' else '.75' for n in names])
            for j, n in enumerate(names): ax.text(j, part.loc[n, 'auroc']+.015, f"{part.loc[n, 'auroc']:.3f}", ha='center', fontsize=7)
            ax.set_xticks(np.arange(len(names)), [n.replace('depth', 'K=') for n in names], rotation=30, ha='right')
            ax.set_title(ds); ax.set_ylabel('AUROC'); ax.set_ylim(0, 1); ax.spines[['top', 'right']].set_visible(False)
        for extension in ['pdf', 'png']: fig.savefig(output/f'{name}.{extension}', dpi=300)
        plt.close(fig)
    return dict(state_effects=state_rows, state_differences=state_differences,
                input_effects=input_rows, transfer=transfer_rows)


def main():
    root = ROOT/'results/hne_latent'; output = root/'summary'; output.mkdir(exist_ok=True)
    config = json.loads((ROOT/'configs/hne_latent.json').read_text())
    rows = []; differences = []; selection = []; scopes = []; seed_rows = []; fold_rows = []; controlled = []; jackknife = []
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'pdf.fonttype': 42})
    for dataset in config['datasets']:
        folds = sorted(int(p.parent.name[4:]) for p in (root/dataset/'recurrent').glob('fold*/held_person.csv'))
        if not folds: continue
        scope = dict(dataset=dataset, completed_folds=folds, complete_five_folds=len(folds)==5)
        scopes.append(scope)
        sources = {k: v for k, v in dataset_sources(dataset).items() if k.startswith(('DNB', 'FCM', 'CATCH', 'Chronos')) or k=='previous_HNE'}
        for name in ['FCM-score-LR', 'CATCH-score-LR']:
            sources[name+' daily'] = (ROOT/'results/public_daily_repeats/ensemble'/dataset/name, None)
        for variant in config['variants']+['detached_feedback', 'frozen_core', 'no_history_coordinates',
                    'no_network_coordinates', 'no_environment_coordinates', 'no_personal_reference']:
            sources[variant] = (root/dataset/variant, None)
        loaded = {name: load(path, folds, col) for name, (path, col) in sources.items()}
        loaded = {name: value for name, value in loaded.items() if value is not None}
        paired = None
        for name, f in loaded.items():
            f = f[KEY+['value']].rename(columns={'value': name})
            paired = f if paired is None else paired.merge(f, on=KEY, validate='one_to_one')
        paired = paired.dropna(subset=list(loaded)); methods = list(loaded)
        paired.to_csv(output/f'{dataset}_paired_predictions.csv', index=False)
        draws = clustered_auc_bootstrap(paired, methods, 2000)
        for j, method in enumerate(methods):
            low, high = np.nanquantile(draws[:, j], [.025, .975])
            ap = sum(len(g)*average_precision_score(g.y, g[method]) for _, g in paired.groupby('fold'))/len(paired)
            rows.append(dict(**scope, method=method, n=len(paired), people=paired.person_id.nunique(),
                             auroc=fold_auc(paired, method), auroc_low=low, auroc_high=high, ap=ap))
            for k, group in paired.groupby('fold'):
                fold_rows.append(dict(dataset=dataset, method=method, fold=int(k), n=len(group), positives=int(group.y.sum()),
                                      auroc=roc_auc_score(group.y, group[method]), ap=average_precision_score(group.y, group[method])))
            if method != 'recurrent':
                low, high = np.nanquantile(draws[:, methods.index('recurrent')]-draws[:, j], [.025, .975])
                differences.append(dict(**scope, comparator=method,
                                        delta=fold_auc(paired, 'recurrent')-fold_auc(paired, method), low=low, high=high))
        controls = [m for m in ['single_pass', 'reset_feedback', 'detached_feedback'] if m in methods]
        if controls:
            effect = np.array([fold_auc(paired, 'recurrent')-fold_auc(paired, m) for m in controls])
            effect_draws = draws[:, methods.index('recurrent'), None]-draws[:, [methods.index(m) for m in controls]]
            radius = np.nanquantile(np.nanmax(np.abs(effect_draws-effect[None]), axis=1), .95)
            for j, method in enumerate(controls):
                controlled.append(dict(**scope, comparator=method, delta=effect[j],
                    simultaneous_low=effect[j]-radius, simultaneous_high=effect[j]+radius,
                    family_size=len(controls), bootstrap_repeats=2000))
                for person in sorted(paired.person_id.unique()):
                    remaining = paired[paired.person_id.ne(person)]
                    jackknife.append(dict(dataset=dataset, comparator=method, omitted_person=person,
                                          delta=fold_auc(remaining, 'recurrent')-fold_auc(remaining, method)))
        for variant in config['variants']+['detached_feedback', 'frozen_core', 'no_history_coordinates',
                    'no_network_coordinates', 'no_environment_coordinates', 'no_personal_reference']:
            for seed in config['seeds']:
                frames = []
                for k in folds:
                    path = root/dataset/variant/f'fold{k}/seed{seed}/held_person.csv'
                    if path.exists(): frames.append(pd.read_csv(path, dtype={'person_id': str}))
                if len(frames)!=len(folds): continue
                frame = pd.concat(frames, ignore_index=True).merge(paired[KEY], on=KEY, validate='one_to_one')
                seed_rows.append(dict(**scope, variant=variant, seed=seed, auroc=fold_auc(frame, 'probability')))
        for seed in [20261110, 20261111, 20261112]:
            baseline_root = (ROOT/'results/unified_daily' if seed==20261110 else
                             ROOT/'results/public_daily_repeats'/f'seed{seed}')
            for method in ['FCM', 'CATCH', 'FCM-score-LR', 'CATCH-score-LR']:
                frame = load(baseline_root/dataset/method, folds)
                if frame is None: continue
                frame = frame.merge(paired[KEY], on=KEY, validate='one_to_one')
                seed_rows.append(dict(**scope, variant=method+' daily', seed=seed, auroc=fold_auc(frame, 'value')))
        external = [m for m in methods if m.startswith(('DNB', 'FCM', 'CATCH', 'Chronos'))]
        strongest = max(external, key=lambda m: fold_auc(paired, m))
        bootstrap_best = draws[:, methods.index('recurrent')]-draws[:, [methods.index(m) for m in external]].max(1)
        low, high = np.nanquantile(bootstrap_best, [.025, .975])
        differences.append(dict(**scope, comparator='strongest_external', strongest_observed=strongest,
            delta=fold_auc(paired, 'recurrent')-fold_auc(paired, strongest), low=low, high=high))
        for variant in config['variants']+['detached_feedback']:
            available_folds = [k for k in config['folds'] if (root/dataset/variant/f'fold{k}/selection.json').exists()]
            if not available_folds: continue
            columns = min(3, len(available_folds)); nrows = int(np.ceil(len(available_folds)/columns))
            fig, axes = plt.subplots(nrows, columns, figsize=(3.25 if columns==1 else 6.75, 2.3*nrows),
                                     layout='constrained', squeeze=False)
            for slot, k in enumerate(available_folds):
                folder = root/dataset/variant/f'fold{k}'; path = folder/'selection.json'
                info = json.loads(path.read_text()); curves = pd.read_csv(folder/'inner_curves.csv').groupby('epoch').mean(numeric_only=True)
                selection.append(dict(dataset=dataset, variant=variant, fold=k, selected_epoch=info['selected_epoch'],
                                      evaluated_epochs=len(info['history']), inner_auroc=info['inner_auroc']))
                ax = axes.flat[slot]
                ax.plot(curves.index, curves.train_bce, color='black', label='Training BCE')
                ax.plot(curves.index, curves.validation_bce, color='.55', label='Validation BCE')
                ax.axvline(info['selected_epoch'], color='.3', linestyle=':', linewidth=.8)
                ax.set_title(f'Fold {k+1}'); ax.set_xlabel('Epoch'); ax.set_ylabel('BCE')
                ax.spines[['top', 'right']].set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if axes.size > len(available_folds):
                for ax in list(axes.flat)[len(available_folds):]: ax.set_axis_off()
                axes.flat[-1].legend(handles, labels, loc='center', frameon=False)
            else: axes.flat[0].legend(handles, labels, loc='best', frameon=False, fontsize=7)
            for suffix in ['pdf', 'png']: fig.savefig(output/f'{dataset}_{variant}_learning.{suffix}', dpi=300)
            plt.close(fig)
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(output/'paired_differences.csv', index=False)
    pd.DataFrame(selection).to_csv(output/'epoch_selection.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(output/'seed_results.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output/'fold_results.csv', index=False)
    pd.DataFrame(controlled).to_csv(output/'rq2_simultaneous_intervals.csv', index=False)
    pd.DataFrame(jackknife).to_csv(output/'rq2_leave_one_person_out.csv', index=False)
    fits = list(root.glob('*/*/fold*/seed*/result.json'))
    other = auxiliary(root, output, config)
    (output/'results.json').write_text(json.dumps(dict(version=config['version'], scope=scopes,
         completed_final_fits=len(fits), metrics=rows, differences=differences, epoch_selection=selection,
         seed_results=seed_rows, fold_results=fold_rows, rq2_simultaneous=controlled,
         **other), indent=2), encoding='utf8')
    if rows:
        table = pd.DataFrame(rows); fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65), layout='constrained')
        order = ['DNB-tuned daily', 'FCM daily', 'CATCH daily', 'Chronos-2 FT', 'previous_HNE', 'single_pass', 'reset_feedback', 'recurrent']
        for ax, dataset in zip(axes, config['datasets']):
            part = table[table.dataset.eq(dataset)].set_index('method'); names = [x for x in order if x in part.index]
            if not names: ax.set_axis_off(); continue
            values = part.loc[names, 'auroc'].to_numpy()
            ax.bar(np.arange(len(names)), values, color=['black' if n=='recurrent' else '.75' for n in names], width=.7)
            for j, value in enumerate(values): ax.text(j, value+.015, f'{value:.3f}', ha='center', fontsize=7)
            ax.set_xticks(np.arange(len(names)), [LABELS.get(n, n.replace(' daily', '')) for n in names], rotation=55, ha='right')
            ax.set_ylim(0, 1.02); ax.set_ylabel('AUROC'); ax.set_title(dataset); ax.spines[['top', 'right']].set_visible(False)
        for suffix in ['pdf', 'png']: fig.savefig(output/f'main_comparison.{suffix}', dpi=300)
        plt.close(fig)
        lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr}', r'\toprule',
            r' & \multicolumn{2}{c}{RESILIENT} & \multicolumn{2}{c}{BrisT1D-Open} \\',
            r'Method & AUROC & AUPRC & AUROC & AUPRC \\', r'\midrule']
        table_order = ['DNB daily', 'DNB-tuned daily', 'FCM daily', 'FCM-score-LR daily',
                       'CATCH daily', 'CATCH-score-LR daily', 'Chronos-2 FT', 'previous_HNE',
                       'single_pass', 'reset_feedback', 'detached_feedback', 'recurrent']
        for method in table_order:
            if method not in table.method.values: continue
            cells = []
            for dataset in config['datasets']:
                part = table[table.dataset.eq(dataset)&table.method.isin(table_order)].set_index('method')
                for metric in ['auroc', 'ap']:
                    if method not in part.index: cells.append('--'); continue
                    ranked = part[metric].sort_values(ascending=False); text = f'{part.loc[method, metric]:.3f}'
                    if ranked.index[0]==method: text = r'\textbf{'+text+'}'
                    elif len(ranked)>1 and ranked.index[1]==method: text = r'\underline{'+text+'}'
                    cells.append(text)
            lines.append(LABELS.get(method, method.replace(' daily', ''))+' & '+' & '.join(cells)+r' \\')
        lines += [r'\bottomrule', r'\end{tabular*}']
        (output/'main_table.tex').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps(dict(completed_final_fits=len(fits), scope=scopes)), flush=True)


if __name__ == '__main__': main()
