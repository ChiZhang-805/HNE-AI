"""Prepare both main datasets from their authors' source archives."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--resilient-archive', type=Path, required=True)
parser.add_argument('--brist1d-archive', type=Path, required=True)
args = parser.parse_args()


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)],
                   cwd=ROOT, check=True)


run('prepare_development_hourly.py', '--dataset', 'RESILIENT', '--source', args.resilient_archive.resolve())
run('prepare_development_hourly.py', '--dataset', 'BrisT1D-Open', '--source', args.brist1d_archive.resolve())
run('build_daily_study.py', '--datasets', 'RESILIENT', 'BrisT1D-Open')
run('build_unified_features.py', '--frequency', 'daily')
run('prepare_hne_routed.py')
