"""Resume the recorded logging failure without changing training computations.

Original manifest/source archive and first-epoch optimizer state are preserved.
The training function must be AST-identical after removing logging expressions;
all other scientific source files, configuration and private inputs stay fixed.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def training_ast(source):
    class WithoutLogging(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'log':
                return None
            return self.generic_visit(node)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_stage')
    return ast.dump(WithoutLogging().visit(function), include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-key', required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    directory = (ROOT / 'artifacts/experiments' / args.run_key).resolve()
    if not directory.is_relative_to((ROOT / 'artifacts/experiments/core').resolve()):
        raise ValueError('Expected a core run key')
    original = read(directory / 'manifest.json'); status = read(directory / 'status.json')
    if status['status'] != 'failed' or 'multiple values for keyword argument' not in status.get('error', ''):
        raise ValueError('This repair only resumes the recorded logging-key collision')
    if (directory / 'continuation.json').exists():
        raise FileExistsError('Continuation already recorded; use the canonical runner with its recorded source commit')
    with zipfile.ZipFile(directory / 'sources.zip') as archive:
        old_source = archive.read('experiments/core/run.py').decode('utf-8')
    runner = ROOT / 'experiments/core/run.py'
    assert training_ast(old_source) == training_ast(runner.read_text(encoding='utf-8'))
    runtime = copy.deepcopy(original)
    runtime['code_commit'] = args.code_commit
    for relative, old_hash in original['source_sha256'].items():
        current = sha(ROOT / relative)
        if relative != 'experiments/core/run.py':
            assert current == old_hash, f'Unrelated source changed: {relative}'
        runtime['source_sha256'][relative] = current
    assert (directory / 'baseline/candidate0_training.pt').is_file()
    checkpoint_hash = sha(directory / 'baseline/candidate0_training.pt')
    receipt = dict(original_manifest_sha256=sha(directory / 'manifest.json'), runtime_manifest=runtime,
                   prior_invocation_seconds=status['seconds'], preserved_failure=status,
                   saved_training_checkpoint_sha256=checkpoint_hash,
                   scientific_training_ast_identical=True, configuration_and_input_contract_unchanged=True,
                   correction='Rename epoch duration in log payload; verify and reuse completed prefix receipts on resume',
                   continuation_tool_sha256=sha(Path(__file__)))
    with zipfile.ZipFile(directory / 'continuation_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in runtime['source_sha256']:
            archive.write(ROOT / relative, relative)
        archive.write(Path(__file__), 'experiments/core/continue_run.py')
    (directory / 'continuation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(training_ast_identical=True, checkpoint_preserved=True, original_manifest_preserved=True)), flush=True)
    raise SystemExit(subprocess.call([sys.executable, str(runner), '--code-commit', args.code_commit]))


if __name__ == '__main__':
    main()
