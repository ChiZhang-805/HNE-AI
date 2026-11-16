"""Fit public baselines on the source pilot and score the paired development folds."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'cache/hydra')]
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import numpy as np
import pandas as pd
import psutil
import torch
from hydra import SparseScaler
from hydra_multivariate import HydraMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from fit_public_trees import tree_scores
from training_data import load_people, selected_origins, provenance_key

TRANSFER = ROOT / 'results/guidance_transfer'
OUT = ROOT / 'results/public_baselines'
KEY = ['dataset', 'person', 'origin']


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_windows(people, protocol, maximum=0):
    history = protocol['history_days']
    samples, rows = [], []
    for person in people:
        with np.load(person.source_files[0], allow_pickle=False) as daily:
            # These arrays use only the fixed 56-day personal reference for scaling.
            # Pool sensor families to give both datasets the same feature meanings.
            state = daily['normalized']
            context = daily['normalized_contexts'][:, 0]
            eligible = daily['reference_eligible'].astype(bool)
            families = daily['state_families']
            names = np.unique(families[eligible])
            if len(names) < 2:
                raise ValueError('A scorable person must have at least two eligible families')
            z = state[:, eligible]
            seen = np.isfinite(z)
            valid_values = np.where(seen, z, 0)
            family_max, family_seen = [], []
            for family in names:
                columns = families[eligible] == family
                family_max.append(np.abs(valid_values[:, columns]).max(axis=1))
                family_seen.append(seen[:, columns].any(axis=1))
            ranked = np.sort(np.stack(family_max, axis=1), axis=1)
            sequence = np.column_stack([
                valid_values.sum(axis=1) / seen.sum(axis=1).clip(min=1),
                ranked[:, -1], ranked[:, -2], np.mean(family_seen, axis=0),
                np.where(np.isfinite(context), context, 0), np.isfinite(context),
            ]).astype('float32')
        for origin, label in selected_origins(person, maximum):
            start = origin + 1 - history
            if start < protocol['reference_days']:
                raise ValueError('Reference overlaps recent history')
            samples.append(sequence[start:origin + 1].T)
            rows.append((person.dataset, person.person, origin, int(label)))
    return np.stack(samples), pd.DataFrame(rows, columns=[*KEY, 'label'])


def tree_features(series):
    features = [series[:, :, -1]]
    for days in (7, 14, 32):
        window = series[:, :, -days:]
        features.extend([window.mean(axis=2), window.std(axis=2), window.max(axis=2)])
    return np.concatenate(features, axis=1)


def guidance_scores(validation, datasets):
    frames = []
    for dataset in datasets:
        path = TRANSFER / (dataset + '_predictions.csv')
        df = pd.read_csv(path, dtype={'person': str}, float_precision='round_trip')
        frames.append(df.assign(dataset=dataset)[[*KEY, 'label', 'aligned']])
    scores = pd.concat(frames).set_index(KEY).sort_index()
    expect = validation.set_index(KEY).sort_index()
    if not scores.index.is_unique or not scores.index.equals(expect.index) or not np.array_equal(scores.label, expect.label):
        raise ValueError('Guidance and public methods must score exactly the same labeled origins')
    return scores.aligned.reindex(pd.MultiIndex.from_frame(validation[KEY])).to_numpy()


def evaluate(frame):
    y = frame.label
    tn, fp, fn, tp = confusion_matrix(y, frame.predicted, labels=[0, 1]).ravel()
    return dict(n=len(y), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=(tn+tp)/len(y), sensitivity=tp/(tp+fn),
                auroc=roc_auc_score(y, frame.score), auprc=average_precision_score(y, frame.score))


def run():
    torch.set_num_threads(1)
    process = psutil.Process()
    if os.name == 'nt':
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start = time.perf_counter()
    cfg = json.loads((ROOT / 'configs/guidance_transfer.json').read_text())
    protocol = json.loads((ROOT / 'configs/daily_protocol.json').read_text())
    source, target = datasets = (cfg['source_dataset'], cfg['target_dataset'])
    fold, seed = cfg['outer_fold'], cfg['seed']
    pilot = json.loads((ROOT / 'configs/key_comparison.json').read_text())['profiles'][cfg['source_profile']]
    train_people = load_people(source, fold, only_split='train')['train']
    train, train_ids = read_windows(train_people, protocol, pilot['train_origins_per_person'])
    val_people = [person for dataset in datasets for person in
                  load_people(dataset, fold, only_split='validation')['validation']]
    val, ids = read_windows(val_people, protocol, cfg['origins_per_person'])
    own_scores = guidance_scores(ids, datasets)
    print(f'Source training origins: {len(train_ids)}. Paired validation origins: {len(ids)}.', flush=True)
    if set(train_ids.person) & set(ids.loc[ids.dataset == source, 'person']):
        raise ValueError('Overlapping source participants')

    frames = []
    def add(name, scores, predicted):
        frames.append(ids.assign(method=name, score=np.asarray(scores), predicted=np.asarray(predicted, dtype=int)))

    tree_train, tree_val = tree_features(train), tree_features(val)
    for name in ('Random Forest', 'XGBoost', 'LightGBM'):
        print('Fitting', name, flush=True)
        p = tree_scores(name, seed, tree_train, train_ids.label.to_numpy(), tree_val)
        add(name, p, p >= .5)

    torch.manual_seed(seed)
    print('Fitting HYDRA', flush=True)
    with torch.inference_mode():
        hydra = HydraMultivariate(train.shape[-1], train.shape[1])
        scaler = SparseScaler()
        x_train = scaler.fit_transform(hydra.batch(torch.from_numpy(train), batch_size=16)).double().numpy()
        x_val = scaler.transform(hydra.batch(torch.from_numpy(val), batch_size=16)).double().numpy()
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
        x_train, train_ids.label, groups=train_ids.person))
    if any(train_ids.label.iloc[tr].nunique() < 2 for tr, _ in folds):
        raise ValueError('A source-only HYDRA selection fold has one training class')
    head = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=folds, scoring='accuracy')
    head.fit(x_train, train_ids.label)
    add('HYDRA', head.decision_function(x_val), head.predict(x_val))
    add('SAGE-TS', own_scores, own_scores >= .5)
    records = pd.concat(frames, ignore_index=True)
    stats = {d: {m: evaluate(f) for m, f in records[records.dataset == d].groupby('method')}
             for d in datasets}
    metadata = dict(source=source, target=target, split='validation', outer_fold=fold, seed=seed,
                    source_training_origins=len(train_ids), source_training_positives=int(train_ids.label.sum()),
                    source_training_people=train_ids.person.nunique(), metrics=stats,
                    binary_decision='Probability >= 0.5; HYDRA uses the zero ridge margin and native class decision.',
                    hydra_upstream='https://github.com/angus924/hydra/tree/144bb7aa3186654042f00f36b078370c620acb06',
                    hydra_alpha=float(head.alpha_), packages={k: version(k) for k in
                    ('torch', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm')},
                    seconds=time.perf_counter()-start, cpu_threads=1)
    inputs = [Path(__file__), ROOT/'scripts/fit_public_trees.py', ROOT/'src/training_data.py', ROOT/'configs/daily_protocol.json',
              ROOT/'configs/key_comparison.json', ROOT/'configs/guidance_transfer.json',
              ROOT/'cache/hydra/hydra.py', ROOT/'cache/hydra/hydra_multivariate.py',
              *(TRANSFER / (d+'_predictions.csv') for d in datasets),
              *(Path(p.source_files[0]) for p in train_people + val_people)]
    metadata['sha256'] = {provenance_key(p, ROOT): hash_file(p) for p in inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        work = Path(tmp)
        records.to_csv(work/'predictions.csv', index=False)
        train_ids.to_csv(work/'training_origins.csv', index=False)
        (work/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf8')
        for file in ('predictions.csv', 'training_origins.csv', 'summary.json'):
            os.replace(work/file, OUT/file)
    print(f'{len(train_ids)} training origins, {len(ids)} paired validation origins, '
          f'{len(frames)} methods, {metadata["seconds"]:.1f} s')


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        run()
