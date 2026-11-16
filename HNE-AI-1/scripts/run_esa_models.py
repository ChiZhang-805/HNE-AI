"""Shared HNE fits and fixed human-source transfer on complete ESA missions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_hne_external as shared
from scripts import run_public_external as public
from scripts.run_transfer import sha, write_json


def configuration(feature_dir, mission, mapping):
    cfg = shared.configuration()
    with np.load(feature_dir / 'train.npz') as z:
        cfg.update(expected_nodes=list(z['nodes'].shape[1:]), expected_window=list(z['windows'].shape[1:]))
    cfg.update(version=f'esa-v2-mission{mission}-20261110', partition_unit='time',
               primary_comparison=['ZeroShot-HNE-additive', 'ZeroShot-HNE-history', 'DNB'],
               secondary_comparison='All target-trained methods are separate from frozen human-source transfer',
               data_mapping=mapping, split=mapping['split'][str(mission)])
    cfg['dnb']['input'] = f"{cfg['expected_nodes'][0]} reference-selected states"
    cfg['mmdev']['input'] = f"{cfg['expected_window'][1]} state/context coordinates"
    cfg['test_bootstrap'] = {'unit': '64-day temporal blocks within one mission', 'replicates': 2000, 'seed': 20261110}
    return cfg


def fit(feature_dir, output, mission, mapping, workers):
    cfg = configuration(feature_dir, mission, mapping)
    shared.fit(feature_dir, output, cfg, workers)


def freeze_models(output):
    cfg = json.loads((output / 'fit_config.json').read_text())['config']
    for method in public.METHODS:
        json.loads((output / method / 'ensemble/selection.json').read_text())
    freeze = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
              'selection_manifest_sha256': sha(output / 'freeze_manifest.json'),
              'source': 'Original frozen RESILIENT+BrisT1D models; no ESA test selection',
              'primary': cfg['primary_comparison'], 'test_constructed': False}
    write_json(output / 'frozen_models.json', freeze)


def analyze(output, mapping):
    frame = pd.read_csv(output / 'test_predictions.csv')
    methods = [*shared.METHODS, 'FCM-3seed', 'CATCH-3seed']
    for method in public.METHODS:
        table = pd.read_csv(output / method / 'ensemble/test.csv')
        frame = frame.merge(table[['person_id', 'origin_day', 'score']].rename(columns={'score':method+'-3seed'}),
                            on=['person_id', 'origin_day'], validate='one_to_one')
    primary = ['ZeroShot-HNE-additive', 'ZeroShot-HNE-history', 'DNB']
    primary_common = np.isfinite(frame[primary]).all(1).to_numpy()
    target_common = np.isfinite(frame[methods]).all(1).to_numpy()
    rows, differences, all_boot, rare_rows, rare_pairs = [], [], {}, [], []
    for population, keep in [('all', np.ones(len(frame), dtype=bool)), ('primary_common', primary_common), ('target_common', target_common)]:
        part = frame.loc[keep]
        if part.empty:
            continue
        blocks = part.origin_day.to_numpy(int)//mapping['bootstrap_block_days']
        n = len(np.unique(blocks))
        draws = np.random.default_rng(mapping['bootstrap_seed']).multinomial(n, np.full(n, 1/n), size=mapping['bootstrap_draws'])
        points, boots, rare_boot = {}, {}, {}
        for method in methods:
            if method.endswith('-3seed'):
                val = pd.read_csv(output / method.removesuffix('-3seed') / 'ensemble/validation.csv')
                threshold = float(np.quantile(val.loc[val.y.eq(0), 'score'], .9))
            else:
                selection = json.loads((output / method / 'selection.json').read_text())
                threshold = selection['threshold_val_negative_p90']
            point, boot, counts = shared.cluster_statistics(part.y.to_numpy(), part[method].to_numpy(), blocks,
                                                            threshold, draws)
            points[method], boots[method] = point, boot
            row = {'population': population, 'method': method, 'origins': counts['origins'],
                   'positive': counts['positive'], 'temporal_blocks': n, 'spacecraft': 1}
            for metric, index in [('auroc', 0), ('ap', 2), ('tpr_val_p90', 3), ('fpr_val_p90', 4)]:
                low, high = np.nanquantile(boot[:, index], [.025, .975])
                row.update({metric: point[index], metric+'_ci_low': low, metric+'_ci_high': high})
            rows.append(row)
            all_boot[population+'__'+method] = boot[:, [0, 2, 3, 4]]
            if 'future_rare_event' in part:
                for subgroup, selected in [('benign_rare', part.y.eq(0) & part.future_rare_event.eq(1)),
                                           ('ordinary', part.y.eq(0) & part.future_rare_event.eq(0))]:
                    valid = selected.to_numpy() & np.isfinite(part[method].to_numpy())
                    _, inverse = np.unique(blocks, return_inverse=True)
                    denominator = np.bincount(inverse[valid], minlength=n)
                    alert = valid & (part[method].to_numpy() > threshold) if threshold is not None else np.zeros(len(part), bool)
                    numerator = np.bincount(inverse[alert], minlength=n)
                    boot_denominator = draws @ denominator
                    rates = np.divide(draws @ numerator, boot_denominator,
                                      out=np.full(len(draws), np.nan), where=boot_denominator > 0)
                    rate = float(numerator.sum()/denominator.sum()) if denominator.sum() else np.nan
                    low, high = np.nanquantile(rates, [.025, .975])
                    rare_boot[(method, subgroup)] = (rate, rates)
                    rare_rows.append({'population': population, 'method': method, 'subgroup': subgroup,
                                      'origins': int(denominator.sum()), 'false_alarms': int(numerator.sum()),
                                      'fpr': rate, 'ci_low': low, 'ci_high': high,
                                      'threshold': threshold})
        if rare_boot:
            for model, control in [('ZeroShot-HNE-additive', 'ZeroShot-HNE-history'),
                                   ('HNE-additive', 'HNE-no-environment'), ('HNE-additive', 'History-HGB')]:
                for subgroup in ['benign_rare', 'ordinary']:
                    value, distribution = rare_boot[(model, subgroup)]
                    base, base_distribution = rare_boot[(control, subgroup)]
                    low, high = np.nanquantile(distribution-base_distribution, [.025, .975])
                    rare_pairs.append({'population': population, 'model': model, 'baseline': control,
                                       'subgroup': subgroup, 'fpr_difference': value-base,
                                       'ci_low': low, 'ci_high': high})
        comparisons = [('ZeroShot-HNE-additive', m) for m in ['ZeroShot-HNE-history', 'DNB']]
        comparisons += [('HNE-additive', m) for m in ['HNE-history', 'HNE-no-network', 'HNE-no-environment', 'History-HGB', 'DNB', 'MM-DEV-daily', 'FCM-3seed', 'CATCH-3seed']]
        for model, control in comparisons:
            delta = boots[model][:, 0]-boots[control][:, 0]
            low, high = np.nanquantile(delta, [.025, .975])
            differences.append({'population': population, 'model': model, 'baseline': control,
                                'difference': points[model][0]-points[control][0], 'ci_low': low, 'ci_high': high})
        controls = ['DNB', 'MM-DEV-daily', 'History-LR', 'History-HGB', 'FCM-3seed', 'CATCH-3seed']
        strongest = max(controls, key=lambda m: points[m][0])
        delta = boots['HNE-additive'][:, 0]-np.max(np.stack([boots[m][:, 0] for m in controls]), axis=0)
        low, high = np.nanquantile(delta, [.025, .975])
        differences.append({'population': population, 'model': 'HNE-additive', 'baseline': 'strongest-target-control',
                            'difference': points['HNE-additive'][0]-points[strongest][0],
                            'ci_low': low, 'ci_high': high, 'strongest_control': strongest})
    pd.DataFrame(rows).to_csv(output / 'test_metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'test_paired_differences.csv', index=False)
    if rare_rows:
        pd.DataFrame(rare_rows).to_csv(output / 'benign_event_false_alarms.csv', index=False)
        pd.DataFrame(rare_pairs).to_csv(output / 'benign_event_paired_differences.csv', index=False)
    np.savez_compressed(output / 'test_bootstrap.npz', **all_boot)
    print(pd.DataFrame(rows).query("population == 'primary_common' and method in @primary").to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missions', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--phase', choices=['fit', 'public-fit', 'freeze', 'predict', 'analysis'], required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--endpoint', choices=['native', 'fluctuation'], default='native')
    args = parser.parse_args()
    mapping = json.loads((ROOT / 'configs/esa_mapping.json').read_text())
    study = 'esa_native' if args.endpoint == 'native' else 'esa_study'
    if args.endpoint == 'native':
        mapping['endpoint'] = json.loads((ROOT / 'configs/esa_native_event.json').read_text())
    for mission in args.missions:
        feature_dir = ROOT / f'data/{study}/ESA-Mission{mission}'
        output = ROOT / f'results/{study}/ESA-Mission{mission}'
        if args.phase == 'fit':
            fit(feature_dir, output, mission, mapping, args.workers)
        elif args.phase == 'public-fit':
            for method in public.METHODS:
                for seed in public.SEEDS:
                    public.fit_one(feature_dir, output, method, seed, partition_unit='time')
                public.ensemble(output, method, ['train', 'validation'])
        elif args.phase == 'freeze':
            freeze_models(output)
        else:
            if args.phase == 'predict':
                freeze = json.loads((output / 'frozen_models.json').read_text())
                shared.predict(feature_dir, output, freeze['config'], args.workers, summarize=False)
                public.predict(feature_dir, output, public.METHODS, public.SEEDS)
                for method in public.METHODS:
                    public.ensemble(output, method, ['test'])
            analyze(output, mapping)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        main()
