"""Check experiment ownership, dependency coverage and public run receipts.

Standard library only. Never starts a run, reads participant outcomes or connects
to a service. Existing historical experiment directories remain untouched.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def project_path(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT), f'Path outside project: {relative}'
    return path


def main():
    registry = read('experiments/registry.json')
    assert registry['project'] == 'HNE-AI-2'
    assert set(registry['roles']) == {'A', 'B'}
    digest = hashlib.sha256(project_path(registry['protocol_file']).read_bytes()).hexdigest()
    assert digest == registry['protocol_sha256'], 'Fixed protocol changed'
    assert set(registry['method_owners']) == {m['id'] for m in registry['planned_method_catalog']}
    assert set(registry['method_owners']) == {f'M{i:02d}' for i in range(1, 14)}
    assert registry['formal_core_units'] == 390
    assert registry['initial_formal_units'] == 990
    active_methods = {m['id'] for m in read('configs/experiment_matrix.json')['core_methods']}
    assert active_methods <= set(registry['method_owners'])
    plans = {}
    bundle_owners = {}
    method_owners = {}
    for entry in registry['packages']:
        topic = entry['id']
        assert topic not in plans and topic not in registry['roles']
        assert entry['path'] == f'experiments/{topic}'
        directory = project_path(entry['path'])
        plan = read(f"{entry['path']}/plan.json")
        assert plan['id'] == topic and plan['owner'] == entry['owner']
        assert plan['owner'] in registry['roles']
        assert plan['reviewer'] in registry['roles'] and plan['reviewer'] != plan['owner']
        assert plan['status'] in registry['status_values']
        assert (directory / 'README.md').is_file() and (directory / 'runs/README.md').is_file()
        assert plan['entry_points'] and plan['acceptance'] and plan['tasks']
        for path in plan['entry_points']:
            assert project_path(path).is_file(), f'Missing entry: {topic}/{path}'
        for path in plan.get('pending_local_entry_points', []):
            project_path(path)  # A planned handoff is not claimed as an existing file.
        task_ids = [task['id'] for task in plan['tasks']]
        assert len(task_ids) == len(set(task_ids))
        assert all(task['status'] in registry['status_values'] for task in plan['tasks'])
        for bundle in plan['paper_bundles']:
            assert bundle not in bundle_owners, f'Duplicate bundle owner: {bundle}'
            bundle_owners[bundle] = topic
        for method in plan['core_methods']:
            assert method not in method_owners, f'Duplicate method owner: {method}'
            method_owners[method] = {'owner': plan['owner'], 'package': topic}
        plans[topic] = plan
    assert method_owners == registry['method_owners']
    assert set(bundle_owners) == set(registry['paper_bundle_ids']), 'Unassigned paper bundle'
    complete = set()

    def visit(topic, stack):
        assert topic in plans, f'Unknown prerequisite: {topic}'
        assert topic not in stack, f'Dependency cycle: {stack + [topic]}'
        if topic in complete:
            return
        for dependency in plans[topic]['depends_on_for_full_runs']:
            visit(dependency, stack + [topic])
        complete.add(topic)

    for topic in plans:
        visit(topic, [])
    receipts = set()
    for topic, plan in plans.items():
        for file in project_path(plan['run_receipts']).glob('*.json'):
            receipt = json.loads(file.read_text(encoding='utf-8'))
            assert not receipt.get('example_only'), f'Template copied as a claim: {file}'
            assert receipt['owner'] == plan['owner']
            assert receipt['status'] in registry['status_values']
            assert receipt['protocol_sha256'] == digest
            key = receipt['run_key']
            parts = key.split('/')
            assert len(parts) == 6 and parts[0] == topic and '<' not in key
            assert key not in receipts, f'Duplicate claimed key: {key}'
            method = parts[4]
            if method in registry['method_owners']:
                assert registry['method_owners'][method]['package'] == topic
            if receipt['status'] != 'planned':
                assert receipt.get('code_commit') and receipt.get('config_sha256')
            if receipt['status'] == 'completed':
                assert receipt.get('shared_inputs_verified') and receipt.get('summary') is not None
                assert receipt.get('elapsed_seconds') is not None
            receipts.add(key)
    for path in [registry['interface_contract'], registry['coordination'], 'paper/README.md',
                 'paper/tables/README.md', 'paper/figures/README.md']:
        assert project_path(path).is_file(), f'Missing contract: {path}'
    print(json.dumps(dict(status='passed',experiment_packages=len(plans),paper_bundles=len(bundle_owners),
                          canonical_methods=len(method_owners),claimed_run_receipts=len(receipts),
                          formal_core_units=390,initial_formal_units=990,training_started=False)))


if __name__ == '__main__':
    main()
