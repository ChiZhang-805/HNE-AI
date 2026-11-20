"""Reconcile measured execution, preserve source snapshots, and audit inputs."""
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import platform
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from study_inputs import SOURCE
sys.path.insert(0,str(ROOT/'cache/public_baseline_packages'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    statuses={name:json.loads((ROOT/'artifacts'/name/'status.json').read_text()) for name in ('local_baselines','cached_evidence','teacher_selection','local_alarms','cached_null_controls','local_model_recovery')}
    assert all(s['status']=='completed' for s in statuses.values())
    assert statuses['local_baselines']['completed_units']==150 and statuses['cached_evidence']['completed_units']==45
    assert statuses['local_alarms']['completed_units']==150
    assert statuses['cached_null_controls']['completed_units']==6
    assert statuses['local_model_recovery']['completed_units']==30
    recovery_errors=[]
    for path in (ROOT/'artifacts/local_model_recovery').glob('*/*/*/report.json'):
        record=json.loads(path.read_text());assert sha(path.parent/'estimator.ubj')==record['model_sha256']
        old=ROOT/'artifacts/local_baselines'/record['dataset']/f"fold{record['fold']}"/f"seed{record['seed']}"/'XGBoost'
        assert sha(old/'estimator.joblib')==record['original_model_sha256']
        assert sha(old/'report.json')==record['original_report_sha256']
        recovery_errors.extend(record['parity'].values());assert record['round_trip_max_error']==0
    assert len(recovery_errors)==60 and max(recovery_errors)<1e-6
    files={}
    for path in (ROOT/'artifacts/local_baselines').glob('*/*/*/*/report.json'):
        for name,expected in json.loads(path.read_text())['data_sha256'].items():
            if name in files and files[name]!=expected:raise AssertionError('Data changed between baseline cells')
            files[name]=expected
    alarm_data=json.loads((ROOT/'artifacts/local_alarms/manifest.json').read_text())['data_sha256']
    for name,expected in alarm_data.items():
        if name in files:assert files[name]==expected
        files[name]=expected
    for name,expected in files.items():assert sha(name)==expected,name
    preserved={'configs/daily_protocol.json':'3d58b3089a3e7d25a442ec8aa78638c509fc03fcf2af901e1cfbe2b5ab17607c',
               'src/protocol.py':'8b2b157145226ee821cec7a965627b6992961702b1dfcfff3f7d51e788998e50',
               'results/tables/guidance_results.pdf':'96625a89fce2915ed3afabb51ca39fb6940a47d6395ab3990357b345b45b5e37',
               'results/tables/guidance_ablations.pdf':'a8b9df241a970867726f5029f1539e3740e7279ac545cf3d668a67de3bd606bc'}
    for name,expected in preserved.items():assert sha(ROOT/name)==expected,name
    contract=ROOT/'artifacts/key_comparison/contract_check/BrisT1D-Open/fold0/seed20261114'
    original=json.loads((contract/'manifest.json').read_text())['original_model_sha256']
    assert sha(SOURCE/'tools/models/chronos-2/model.safetensors')==original
    snapshots={}
    for study in statuses:
        directory=ROOT/'artifacts'/study;manifest=directory/'manifest.json';data=json.loads(manifest.read_text())
        source=data.get('source_sha256',data.get('files',{}))
        snapshot=directory/'sources.zip'
        if not snapshot.exists():
            with zipfile.ZipFile(snapshot,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(manifest,'manifest.json')
                for name,expected in source.items():
                    path=ROOT/name
                    if path.suffix not in ('.py','.json'):continue
                    assert sha(path)==expected,name
                    archive.write(path,name.replace('\\','/'))
        snapshots[study]=dict(path=str(snapshot.relative_to(ROOT)),sha256=sha(snapshot))
    packages={}
    for package in ('torch','numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','matplotlib','chronos-forecasting'):
        try:packages[package]=version(package)
        except PackageNotFoundError:packages[package]='not_found_in_metadata'
    out=ROOT/'results/local_experiments'
    ledger=dict(status='local_execution_batch_completed; full_paper_not_completed',
                completed=dict(public_and_HGB_cv_units=150,classifier_candidate_fits=900,conditional_evidence_units=45,conditional_head_candidate_fits=270,
                               teacher_learning_rate_candidates=3,teacher_component_candidates=9,teacher_fitting_note='Each component includes inner selection and source-pool refit; not only nine optimizer fits',
                               full_pretrained_contract_stages=8,full_pretrained_interface_checks=24,batched_head_gradient_checks=15,forecast_gap_settings=4,chronological_baseline_replays=150,no_evidence_head_units=6,no_evidence_candidate_fits=36,serialization_recovery_refits=30),
                statuses=statuses,contract_timing=json.loads((contract/'timing.json').read_text()),source_snapshots=snapshots,
                preserved_inputs=dict(protocol_and_prior_table_hashes=preserved,first_paper_data_files_verified=len(files),original_model_sha256=original),
                runtime=dict(python=platform.python_version(),packages=packages,device='cpu',maximum_training_threads=2,priority='below_normal',api_requests=0,gpu_allocations=0),
                remaining=[
                    dict(work='Full-fold S/L/G and complete interfaces, six-candidate matching, five folds and three seeds',dependency='Dedicated compute/GPU recommended; technically CPU-capable but not measured at full-fold scale; no API required',status='pending'),
                    dict(work='H/N/E retraining, ordinary-statistical/structural guidance, time-qualified retrieval',dependency='Additional implementation and matched refits; no API required',status='pending'),
                    dict(work='Utility gate and harmless fallback',dependency='Entire-pipeline person OOF, including S/LoRA/head/G selection exclusions; not pilot-cache labels',status='pending'),
                    dict(work='External air-quality target validation/test, synthetic mechanisms, corruption and drift',dependency='Frozen framework and selection; preserve external outcome lock until that point',status='pending'),
                    dict(work='Longer forecast gaps',dependency='Population audit complete; independent models and complete issue-time replay still required',status='partially_completed')],
                failures=dict(xgboost_snapshot_loading='Fresh-process native load fails under CUDA_VISIBLE_DEVICES=-1 and succeeds under empty visibility; replay now sets empty visibility. Original snapshots preserved; 30 selected models refitted without reselection, all saved probabilities checked, native UBJSON round trip passed; underlying native cause not diagnosed',
                              recovery_max_prediction_error=max(recovery_errors),replay_failure_archive='artifacts/local_alarms/failed_attempt.zip',
                              cached_head_wall_time_outlier='Final head unit took 1207.21 seconds; cause unconfirmed; included without adjustment in reported 3125.29-second batch'),
                interpretation='Both human cohorts development-exposed. Conditional heads share one pilot S/L; three head seeds are not three pipeline seeds. No internal-guidance superiority or clinical validity established.')
    (out/'execution_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(dict(status=ledger['status'],verified_data_files=len(files),source_snapshots=snapshots)))


if __name__=='__main__':main()
