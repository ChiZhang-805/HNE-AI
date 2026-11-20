"""Source-inner teacher selection and excluded-person conditional-information audit."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.stats import t
from conditional_teacher import HistoryPredictor, ResidualConditionalPredictor, teacher_from_checkpoint
from precursor_guidance import ConditionalPredictor, patch_nll
from train_evidence import predict, forecast_loss, family_mean, calibrate_person, adapt_prediction
from train_comparison import sha, save_json, resource_check
from training_data import Person, FAMILIES, component_assignments
from study_inputs import SOURCE


def reference_people(dataset, fold):
    """Read fold/identity and observations only, never event labels or origin eligibility."""
    groups = {'train': [], 'validation': [], 'evaluation': []}
    excluded = 0
    for daily in sorted((SOURCE / 'data/daily_study' / dataset).glob('*_daily.npz')):
        with np.load(daily, allow_pickle=False) as data:
            person_fold, name = int(data['fold']), str(data['person_id'])
        split = 'evaluation' if person_fold == fold else 'validation' if person_fold == (fold + 1) % 5 else 'train'
        if split == 'evaluation':
            continue
        hourly = SOURCE / 'data/development' / dataset / daily.name.replace('_daily', '_hourly')
        with np.load(hourly, allow_pickle=False) as data:
            raw, clock = data['values'].copy().astype('float32'), pd.Timestamp(str(data['clock']))
        if len(raw) < 56 * 24:
            excluded += 1
            continue
        means, scales, usable = [], [], []
        for channel in raw[:28 * 24].T:
            values = channel[np.isfinite(channel)].astype('float64')
            enough = len(values) >= 8
            mean = np.median(values) if enough else 0.
            sd = np.std(values) if enough else 0.
            good = enough and sd > 1e-6
            scale = max(1.4826 * np.median(np.abs(values - mean)), .1 * sd, 1e-6) if good else 1.
            means.append(mean); scales.append(scale); usable.append(good)
        person = Person(dataset, name, person_fold, raw, clock, np.array([], dtype=int),
                        np.array([], dtype='float32'), np.asarray(means), np.asarray(scales),
                        np.asarray(usable), (str(daily), str(hourly)))
        observed = person.small_window(28, 56)['observed'][0, :5, 21:]
        if not (observed.sum(-1) >= 8).any():
            excluded += 1
            continue
        groups[split].append(person)
    return groups, excluded


def forecast_score(model, people):
    losses = []
    with torch.no_grad():
        for person in people:
            parts = []
            for start, end in ((14, 42), (28, 56)):
                prediction, window = predict(model, person, start, end)
                loss, valid = patch_nll(window['values'], window['observed'], *prediction['views']['HNE'])
                mask = valid[0] & window['state'][0, :, None]
                mask[:, :21] = False
                parts.append(family_mean(loss[0], mask, window['family'][0]))
            losses.append(float(torch.stack(parts).mean()))
    return float(np.mean(losses))


def epoch_fit(model, people, optimizer, seed, epoch):
    model.train()
    for index in np.random.default_rng(seed + epoch).permutation(len(people)):
        resource_check('cpu')
        optimizer.zero_grad()
        loss = sum(forecast_loss(*predict(model, people[index], a, b)) for a, b in ((14, 42), (28, 56))) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite teacher forecast objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()


def train_trial(initial, fit, hold, config, rate, seed, allow_zero):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate, weight_decay=.0001)
    best, selected, stale, curve, state = float('inf'), 0, 0, [], None
    if allow_zero:
        best, state = forecast_score(model, hold), copy.deepcopy(model.state_dict())
        curve.append(dict(epoch=0, heldout_hne_nll=best))
    start = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        epoch_fit(model, fit, optimizer, seed, epoch)
        model.eval()
        score = forecast_score(model, hold)
        curve.append(dict(epoch=epoch, heldout_hne_nll=score))
        if score < best - 1e-7:
            best, selected, stale, state = score, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(state)
    return model, dict(rate=rate, selected_epochs=selected, heldout_hne_nll=best,
                       elapsed_seconds=time.perf_counter() - start, curve=curve)


def refit(initial, people, selection, seed):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=selection['rate'], weight_decay=.0001)
    start = time.perf_counter()
    for epoch in range(1, selection['selected_epochs'] + 1):
        epoch_fit(model, people, optimizer, seed, epoch)
    return model.eval(), time.perf_counter() - start


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_person(model, person, architecture, component, split, config, calibration_policy='independent'):
    calibration = calibrate_person(model, person, config['personal_steps'])
    if calibration_policy == 'shared_H':
        calibration['bias'][1:] = calibration['bias'][0].clone()
        calibration['log_scale'][1:] = calibration['log_scale'][0].clone()
    elif calibration_policy != 'independent':
        raise ValueError(calibration_policy)
    rows = []
    periods = [('reference_score', 28, 56)]
    if len(person.raw) >= 88 * 24:
        periods.append(('fixed_monitoring_day87', 60, 88))
    for period, start, end in periods:
        window = person.small_window(start, end)
        modes = ['aligned', 'older_conditions'] if architecture == 'residual' else ['aligned']
        for mode in modes:
            kwargs = dict(condition_lag=config['condition_lag_patches']) if mode == 'older_conditions' else {}
            with torch.no_grad():
                prediction = model(**window, reference=calibration['reference'], **kwargs)
            for calibrated in (False, True):
                current = adapt_prediction(prediction, calibration['bias'], calibration['log_scale']) if calibrated else prediction
                for view in ('H', 'HN', 'HNE'):
                    mu, scale = (value[0, :, 21:] for value in current['views'][view])
                    x, mask = window['values'][0, :, 21:], window['observed'][0, :, 21:]
                    mask = mask & (mask.sum(-1) >= 8)[..., None] & person.state[0, :, None, None]
                    z = ((x - mu) / scale).detach().numpy()
                    nll = -torch.distributions.StudentT(5., mu, scale).log_prob(x).detach().numpy()
                    families = {}
                    for channel in range(5):
                        valid = mask[channel].numpy()
                        if valid.any():
                            values = z[channel][valid]
                            families.setdefault(int(person.family[0, channel]), []).append([
                                float(nll[channel][valid].mean()), float((np.abs(values) <= t.ppf(.95, 5)).mean()),
                                float(t.cdf(values, 5).mean())])
                    if not families:
                        continue
                    mean = np.mean([np.mean(values, axis=0) for values in families.values()], axis=0)
                    rows.append(dict(dataset=person.dataset, split=split, person=person.person, component=component,
                                     architecture=architecture, period=period, calibrated=calibrated, mode=mode,
                                     view=view, nll=float(mean[0]), coverage90=float(mean[1]), pit_mean=float(mean[2])))
    return rows


def summarize(rows, seed, repetitions):
    frame = pd.DataFrame(rows)
    summary = []
    for key, part in frame.groupby(['dataset', 'split', 'period', 'calibrated', 'architecture', 'mode']):
        pivot = part.pivot(index='person', columns='view', values='nll').dropna()
        item = dict(zip(['dataset', 'split', 'period', 'calibrated', 'architecture', 'mode'], key))
        item['calibrated'] = bool(item['calibrated'])
        item.update(people=len(pivot), nll_H=float(pivot.H.mean()), nll_HN=float(pivot.HN.mean()), nll_HNE=float(pivot.HNE.mean()))
        rng = np.random.default_rng(seed)
        for name, values in [('gain_N', (pivot.H - pivot.HN).to_numpy()), ('gain_E', (pivot.HN - pivot.HNE).to_numpy()),
                             ('gain_NE', (pivot.H - pivot.HNE).to_numpy())]:
            boot = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
            item[name] = dict(mean=float(values.mean()), interval=np.quantile(boot, [.025, .975]).tolist())
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    relative = Path('experiments/teacher') / config['dataset'] / f"fold{config['fold']}" / f"seed{config['seed']}" / 'conditional_teacher' / config_hash[:16]
    directory, output = ROOT / 'artifacts' / relative, ROOT / 'results' / relative
    directory.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    people, excluded = reference_people(config['dataset'], config['fold'])
    assignments = component_assignments(people, config['seed'])
    source_files = [Path(__file__).resolve(), args.config, *sorted((ROOT / 'src').rglob('*.py'))]
    manifest = dict(config=config, config_sha256=config_hash, code_commit=args.code_commit,
                    protocol_sha256=sha(ROOT / 'configs/daily_protocol.json'),
                    source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
                    data_sha256={name: sha(name) for group in people.values() for p in group for name in p.source_files},
                    small_component=assignments, split_people={k: [p.person for p in v] for k, v in people.items()},
                    excluded_for_reference_coverage=excluded, event_labels_loaded=False,
                    selection_exclusion='Each scored training person excluded from fitting, epoch/lr and architecture selection')
    path = directory / 'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Teacher provenance changed; preserve existing experiment')
    save_json(path, manifest)
    if (directory / 'status.json').exists() and json.loads((directory / 'status.json').read_text())['status'] == 'completed':
        print(json.dumps(dict(status='completed', action='preserved_existing_run', directory=str(directory))))
        return
    if not (directory / 'sources.zip').exists():
        with zipfile.ZipFile(directory / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in source_files:
                archive.write(file, file.relative_to(ROOT).as_posix())
    started = time.perf_counter()
    save_json(directory / 'status.json', dict(status='running', pid=os.getpid(), phase='teacher_fit'))
    def log(event, **values):
        row = dict(event=event, seconds=time.perf_counter() - started, **values)
        with (directory / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    records, chosen, audit = [], [], []
    try:
        for component in range(3):
            pool = [p for p in people['train'] if assignments[p.person] != component]
            score_people = [p for p in people['train'] if assignments[p.person] == component]
            pool.sort(key=lambda p: hashlib.sha256(f"{config['seed']}|inner|{p.person}".encode()).hexdigest())
            fit, hold = pool[:-config['inner_holdout_people']], pool[-config['inner_holdout_people']:]
            if not fit or not hold or not score_people:
                raise ValueError('Insufficient independent teacher fitting/selection/scoring people')
            seed = config['seed'] + 101 * component
            selected_fit_history = history_refit = None
            selections = []
            for architecture in config['architectures']:
                torch.manual_seed(seed)
                initial = {'history': HistoryPredictor, 'shared': ConditionalPredictor}.get(architecture, lambda: ResidualConditionalPredictor(copy.deepcopy(selected_fit_history)))()
                trials, trial_states = [], []
                for candidate, rate in enumerate(config['learning_rates']):
                    target = directory / f'component{component}' / architecture / f'candidate{candidate}.pt'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        saved = torch.load(target, map_location='cpu', weights_only=False)
                        fitted = copy.deepcopy(initial); fitted.load_state_dict(saved['state']); result = saved['selection']
                    else:
                        fitted, result = train_trial(initial, fit, hold, config, rate, seed, architecture == 'residual')
                        torch.save(dict(architecture=architecture, state=fitted.state_dict(), selection=result), target)
                    trials.append(result); trial_states.append(copy.deepcopy(fitted.state_dict()))
                    records.append(dict(component=component, architecture=architecture, candidate=candidate,
                                        **{k: result[k] for k in ('rate', 'selected_epochs', 'heldout_hne_nll', 'elapsed_seconds')}))
                    log('candidate_complete', **records[-1])
                best = min(range(len(trials)), key=lambda i: (trials[i]['heldout_hne_nll'], i))
                if architecture == 'history':
                    selected_fit_history = copy.deepcopy(initial); selected_fit_history.load_state_dict(trial_states[best])
                torch.manual_seed(seed)
                refit_initial = ResidualConditionalPredictor(copy.deepcopy(history_refit)) if architecture == 'residual' else initial
                target = directory / f'component{component}' / architecture / 'selected.pt'
                if target.exists():
                    saved = torch.load(target, map_location='cpu', weights_only=False)
                    model = teacher_from_checkpoint(saved)
                else:
                    model, seconds = refit(refit_initial, pool, trials[best], seed)
                    report = dict(component=component, fit_people=[p.person for p in pool],
                                  selection_holdout=[p.person for p in hold], scored_people=[p.person for p in score_people],
                                  selected_epochs=trials[best]['selected_epochs'], refit_seconds=seconds,
                                  selection=trials[best], event_labels_used=False)
                    saved = dict(architecture=architecture, state=model.state_dict(), report=report)
                    torch.save(saved, target)
                if architecture == 'history':
                    history_refit = copy.deepcopy(model)
                model.eval().requires_grad_(False)
                selections.append(dict(architecture=architecture, criterion=trials[best]['heldout_hne_nll'],
                                       path=target.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(target)))
                for person in score_people:
                    assert person.person not in saved['report']['fit_people']
                    assert person.person not in saved['report']['selection_holdout']
                    audit.extend(audit_person(model, person, architecture, component, 'source_oof', config))
                log('architecture_complete', component=component, architecture=architecture,
                    scored_people=len(score_people), selection_nll=trials[best]['heldout_hne_nll'])
                write_csv(output / 'person_metrics.csv', audit)
                write_csv(output / 'candidate_metrics.csv', records)
            best = min(selections, key=lambda x: (x['criterion'], config['architectures'].index(x['architecture'])))
            selected_dir = directory / 'selected'; selected_dir.mkdir(exist_ok=True)
            saved = torch.load(ROOT / best['path'], map_location='cpu', weights_only=False)
            saved['architecture_selection'] = dict(candidates=selections, selected=best['architecture'], selection_uses_scored_people=False)
            target = selected_dir / f'component{component}.pt'; torch.save(saved, target)
            chosen.append(dict(component=component, architecture=best['architecture'], path=target.relative_to(ROOT).as_posix(), sha256=sha(target)))
            log('component_selected', component=component, architecture=best['architecture'])
        # Source validation people never participate in any S fitting/selection.
        for person in people['validation']:
            component = assignments[person.person]
            checkpoint = torch.load(ROOT / chosen[component]['path'], map_location='cpu', weights_only=False)
            assert person.person not in checkpoint['report']['fit_people']
            assert person.person not in checkpoint['report']['selection_holdout']
            audit.extend(audit_person(teacher_from_checkpoint(checkpoint), person, checkpoint['architecture'], component, 'source_validation', config))
        write_csv(output / 'person_metrics.csv', audit)
        result = dict(status='completed', config=config, selected=chosen, seconds=time.perf_counter() - started,
                      source_training_people=len(people['train']), source_validation_people=len(people['validation']),
                      conditional_gains=summarize(audit, config['seed'], config['bootstrap_repetitions']),
                      uncertainty='Person bootstrap conditional on fitted overlapping source models; not retraining uncertainty',
                      scope=config['scope'], event_labels_loaded=False, gpu_used=False)
        save_json(output / 'summary.json', result)
        save_json(directory / 'selection.json', dict(selected=chosen, manifest_sha256=sha(directory / 'manifest.json')))
        save_json(directory / 'status.json', dict(status='completed', seconds=result['seconds']))
        log('completed', selected_architectures=[x['architecture'] for x in chosen], output=str(output))
    except Exception as error:
        save_json(directory / 'status.json', dict(status='failed', error=str(error), seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
