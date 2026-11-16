"""Export actual completed-run metrics, paired differences and learning curves."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys
import zipfile
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_comparison import metrics, save_json, sha, paired_bootstrap

LABELS = {'M04_C2_LoRA': 'No guidance', 'M10_SAGE_fixed': 'Internal guidance',
          'M05_S_logit_residual': 'Evidence + logit', 'M12_S_hidden_late_fusion': 'Evidence + hidden',
          'M12_full_evidence_fusion': 'Full evidence + hidden'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='cpu_pilot')
    parser.add_argument('--dataset', default='BrisT1D-Open')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    directory = ROOT / 'artifacts/key_comparison' / args.profile / args.dataset / f'fold{args.fold}' / f'seed{args.seed}'
    out = ROOT / 'results/key_comparison' / args.profile / args.dataset
    prefix = f'fold{args.fold}_seed{args.seed}'
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'completed':
        raise RuntimeError('Do not summarize a partial run as completed')
    manifest = json.loads((directory / 'manifest.json').read_text())
    # One stable executable-source snapshot per completed run, without creating
    # duplicate working source trees or copying shared data/large model weights.
    snapshot = directory / 'sources.zip'
    if not snapshot.exists():
        for relative, expected in manifest['source_sha256'].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError(f'Cannot snapshot changed training source: {relative}')
        with zipfile.ZipFile(snapshot, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in manifest['source_sha256']:
                archive.write(ROOT / relative, relative)
            archive.write(directory / 'manifest.json', 'manifest.json')
            archive.write(ROOT / 'configs/key_comparison.json', 'configs/key_comparison.json')
            archive.write(ROOT / 'configs/daily_protocol.json', 'configs/daily_protocol.json')
            if (directory / 'runtime.json').exists():
                archive.write(directory / 'runtime.json', 'runtime.json')
            for relative in ('src/fusion/__init__.py', 'src/fusion/evidence.py', 'scripts/train_evidence_fusion.py',
                             'scripts/benchmark_key_comparison.py', 'scripts/summarize_key_comparison.py'):
                archive.write(ROOT / relative, relative)
    result = json.loads((out / f'{prefix}_metrics.json').read_text())
    predictions = pd.read_csv(out / f'{prefix}_predictions.csv')
    timing = json.loads((out / f'{prefix}_timing.json').read_text())
    paired = json.loads((out / f'{prefix}_paired.json').read_text())
    primary = {}
    guide_rows = predictions[(predictions.method == 'M10_SAGE_fixed') & (predictions.split == 'validation')].to_dict('records')
    for comparator in ('M04_C2_LoRA', 'M05_S_logit_residual', 'M12_full_evidence_fusion'):
        if comparator not in result:
            continue
        comparison_rows = predictions[(predictions.method == comparator) & (predictions.split == 'validation')].to_dict('records')
        contrast = paired_bootstrap({'M04_C2_LoRA': {'validation': comparison_rows},
                                     'M10_SAGE_fixed': {'validation': guide_rows}},
                                    args.seed, manifest['config']['bootstrap_repetitions'])['validation/M10_SAGE_fixed']
        primary[f'M10_SAGE_fixed-minus-{comparator}'] = contrast
    save_json(out / f'{prefix}_primary_contrasts.json', primary)
    rows = []
    per_person = []
    differences = []
    for method, report in result.items():
        row = dict(method=method, selected_epoch=report['selected_epoch'], trainable_parameters=report['trainable_parameters'],
                   training_and_validation_seconds=report['training_and_validation_seconds'], **report['validation'])
        rows.append(row)
        df = predictions[(predictions['method'] == method) & (predictions['split'] == 'validation')]
        for person, values in df.groupby('person'):
            per_person.append(dict(method=method, person=person, **metrics(values.to_dict('records'))))
        base = predictions[(predictions['method'] == 'M04_C2_LoRA') & (predictions['split'] == 'validation')]
        matched = df.merge(base, on=['person', 'origin', 'split'], suffixes=('', '_baseline'), validate='one_to_one')
        delta = matched.probability - matched.probability_baseline
        differences.append(dict(method=method, max_absolute_probability_change=float(np.abs(delta).max()),
                                mean_absolute_probability_change=float(np.abs(delta).mean()),
                                changed_ranking_auroc=row['auroc'] - result['M04_C2_LoRA']['validation']['auroc']))
    pd.DataFrame(rows).to_csv(out / f'{prefix}_summary.csv', index=False)
    pd.DataFrame(per_person).to_csv(out / f'{prefix}_per_person.csv', index=False)
    save_json(out / f'{prefix}_prediction_changes.json', differences)
    # Add issue/target times and explicit checkpoint provenance to a separate enriched
    # export, retaining the exact raw predictions from the training program.
    from training_data import load_people, selected_origins
    people = load_people(args.dataset, args.fold, manifest['config']['include_evaluation'])
    lookup = {p.person: p for group in people.values() for p in group}
    prevalence = float(np.mean([np.mean([y for _, y in selected_origins(p, manifest['config']['train_origins_per_person'])]) for p in people['train']]))
    prior_records = [{**r, 'probability': prevalence} for r in guide_rows]
    training_prior = dict(probability=prevalence, validation=metrics(prior_records),
                          scope='Person-balanced training prevalence only; no validation labels used to fit this constant')
    save_json(out / f'{prefix}_training_prior.json', training_prior)
    enriched = predictions.copy()
    for column, offset in [('issue_time', 1), ('target_start', 2), ('target_end_exclusive', 9)]:
        enriched[column] = [str(lookup[p].clock + pd.Timedelta(days=int(o) + offset)) for p, o in zip(enriched.person, enriched.origin)]
    enriched['outer_evaluation_fold'] = args.fold
    enriched['person_fold'] = [lookup[p].fold for p in enriched.person]
    enriched['seed'] = args.seed
    hashes = {m: sha(directory / f'{m}.pt') for m in result}
    enriched['checkpoint_sha256'] = enriched.method.map(hashes)
    enriched.to_csv(out / f'{prefix}_forecast_records.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, report in result.items():
        curve = report['curve']
        epochs = [r['epoch'] for r in curve]
        axes[0].plot(epochs, [r['train_loss'] for r in curve], marker='.', label=LABELS[method])
        axes[1].plot(epochs, [r['validation']['nll'] for r in curve], marker='.', label=LABELS[method])
    axes[0].set(title='Training BCE (person-balanced)', xlabel='Epoch', ylabel='Loss')
    axes[1].set(title='Validation NLL (used for selection)', xlabel='Epoch', ylabel='NLL')
    axes[1].legend(fontsize=8)
    fig.suptitle(f'{args.dataset} / {args.profile}: exploratory development run', fontsize=11)
    fig.savefig(out / f'{prefix}_learning_curves.png', dpi=180)
    plt.close(fig)
    table = pd.DataFrame(rows)[['method', 'auroc', 'auprc', 'brier', 'nll', 'selected_epoch', 'training_and_validation_seconds']]
    table_html = table.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    epoch_times = [r['elapsed_seconds'] for r in result['M04_C2_LoRA']['curve']]
    info = json.dumps(manifest['config'], ensure_ascii=False, indent=2)
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>关键对照实测</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.6 sans-serif;color:#1b2530}}table{{border-collapse:collapse;font-size:14px}}th,td{{padding:8px;border:1px solid #ddd}}pre{{background:#f4f6f8;padding:16px;overflow:auto}}img{{max-width:100%}}</style>
<h1>关键对照实测：{html.escape(args.dataset)}</h1>
<p>用途：{html.escape(args.profile)}。本表来自已完成的实际训练；第一篇数据与原权重只读，当前运行设备为{html.escape(manifest['config']['device'])}。</p>
<p>验证集用于checkpoint选择，不是独立测试。只有{rows[0]['people']}名验证人员、{rows[0]['origins']}个重叠起点和{rows[0]['positives']}个阳性。不能据此宣称内部指导有效或具有顶会级证据。</p>
{table_html}
<p>训练人员等权阳性率的常数预测（不使用验证结局拟合）：概率{prevalence:.4f}，验证NLL {training_prior['validation']['nll']:.6f}、Brier {training_prior['validation']['brier']:.6f}。它用于检查复杂模型的概率误差是否已优于简单参照。</p>
<p>公共S交叉拟合：{timing['small_cross_fit_seconds']:.2f}秒；个人适配与证据：{timing['personal_fit_and_evidence_seconds']:.2f}秒；冻结前缀缓存：{timing['prefix_cache_seconds']:.2f}秒。各方法训练列包含每轮验证，融合头训练复用了冻结特征，因此不包含共同基线/S的训练成本。</p>
<p>本次调用墙钟：{timing['invocation_wall_seconds'] / 60:.2f}分钟；进程峰值内存：{timing['process_peak_rss_bytes'] / 1024 ** 3:.2f}GiB。缓存推理时间不是完整部署时延，完整时延单列于deployment_timing.json。</p>
<p>基线单轮训练加验证：中位数{np.median(epoch_times):.2f}秒，最长{max(epoch_times):.2f}秒。长间隔期间工具连接曾中断，其原因未确证，原始墙钟记录完整保留，不静默扣除。完整证据融合补充运行另用{timing.get('full_evidence_fusion_additional_wall_seconds', 0) / 60:.2f}分钟。</p>
<img src="{prefix}_learning_curves.png" alt="训练与验证曲线">
<p>各曲线是各自阶段的epoch：指导/融合从已训练的基线开始，并非与基线从同一总训练步开始。预先指定仅第{manifest['config']['minimum_selection_epoch']}轮起选checkpoint，早期更低的验证损失不会改写这条规则；因此本pilot不能声称已经充分选型、得到最强基线。</p>
<p>输出修正和摘要隐藏融合使用同一S缓存的12维摘要；完整证据融合另接收所有逐patch节点和逐边证据、掩码与时间位置，并通过可训练编码器汇聚。完整证据融合897参数，内部指导866参数，容量相近但不宣称严格等参。正式实验仍需充分训练基线、更多人员折和其他机制对照。</p>
<h2>实际运行配置</h2><pre>{html.escape(info)}</pre>
<h2>按人配对重采样</h2><p>置信区间仅作探索诊断；验证集参与选型，人数过少，有些重采样不含阳性。</p><pre>{html.escape(json.dumps(paired,ensure_ascii=False,indent=2))}</pre>
<h2>预先指定的主要比较</h2><pre>{html.escape(json.dumps(primary,ensure_ascii=False,indent=2))}</pre>
<h2>相对基线预测变化</h2><pre>{html.escape(json.dumps(differences,ensure_ascii=False,indent=2))}</pre></html>'''
    (out / f'{prefix}_report.html').write_text(body, encoding='utf8')
    print(json.dumps(dict(summary=rows, prediction_changes=differences, report=str(out / f'{prefix}_report.html')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
