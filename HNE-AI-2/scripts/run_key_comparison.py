"""Run/resume the actual pretrained-model key comparisons without modifying paper one."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['pipeline_check', 'cpu_pilot', 'full'], default='cpu_pilot')
    parser.add_argument('--dataset', choices=['BrisT1D-Open', 'RESILIENT'], default='BrisT1D-Open')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--include-evaluation', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/key_comparison.json').read_text(encoding='utf8'))
    cfg = {**config['common'], **config['profiles'][args.profile], **vars(args), 'run_name': args.profile}
    directory = ROOT / 'artifacts/key_comparison' / args.profile / args.dataset / f'fold{args.fold}' / f'seed{args.seed}'
    status_path = directory / 'status.json'
    if status_path.exists():
        recorded = json.loads(status_path.read_text(encoding='utf8'))
        if recorded['status'] == 'completed':
            if recorded['config'] != cfg:
                parser.error('Existing completed run has a different configuration and is preserved')
            print(json.dumps(dict(status='completed', action='no_op_existing_run_preserved',
                                  directory=str(directory), provenance='See the original manifest and sources.zip'), ensure_ascii=False))
            return
    # Only this process is affected; no persistent environment or shared dependencies change.
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = str(cfg['threads'])
    os.environ['MKL_NUM_THREADS'] = str(cfg['threads'])
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    sys.path.insert(0, str(ROOT / 'src'))
    from train_comparison import execute
    execute(ROOT, cfg)


if __name__ == '__main__':
    main()
