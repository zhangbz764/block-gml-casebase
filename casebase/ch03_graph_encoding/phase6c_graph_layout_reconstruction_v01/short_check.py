"""Bounded development check: <=8 Graph reads, 4 total optimizer steps, <=300s.

Not a formal dataset, training run or evidence of reconstruction quality.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
import numpy as np
from data_adapter import (ROOT, adapt_graph, exposure_ids, fit_normalization, load_config, normalize_sample,
                          permute_sample, read_graph, sha256, stable_hash, to_metric, true, write_json)
from prepare_dataset import check_registry, code_hash, environment_info


def run(config):
    started = time.perf_counter(); report = {'state': 'data_check_only_model_runtime_pending', 'formal_prepare_run': False,
        'formal_training_run': False, 'formal_evaluation_run': False, 'optimizer_steps': 0, 'real_cases': []}
    environment = environment_info(); check_registry(config)
    report.update(config_hash=stable_hash(config), code_hash=code_hash())
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    report['unit_checks'] = {'run': outcome.testsRun, 'failures': len(outcome.failures), 'errors': len(outcome.errors),
                             'skipped': len(outcome.skipped), 'skip_reasons': [reason for _, reason in outcome.skipped]}
    if not outcome.wasSuccessful(): raise RuntimeError('Unit checks failed; real short check not started')
    rows = []
    with gzip.open(config['index'], 'rt', encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['case_status']=='completed' and true(row['audit_passed']) and 8 <= int(float(row['node_count'])) <= 64:
                rows.append(row)
            if len(rows)==8: break
    samples = []
    for row in rows:
        if time.perf_counter()-started > 290: raise RuntimeError('Short-check time budget reached')
        info = {'sample_id': row['sample_id'], 'city': row['city'], 'node_count': row['node_count']}
        try:
            record = read_graph(row, config['graph_root']); sample = adapt_graph(record)
            if not np.allclose((to_metric(sample['target'][:, :2], sample['frame'])-sample['frame']['center'])/sample['frame']['L'], sample['target'][:, :2]):
                raise ValueError('Coordinate round trip failed')
            permutation = np.arange(len(sample['node']))[::-1]
            permuted = permute_sample(sample, permutation)
            if not np.allclose(permuted['target'], sample['target'][permutation]): raise ValueError('Target permutation failed')
            samples.append(sample); info['status'] = 'data_adapter_passed'
        except (ValueError, OSError, KeyError) as error: info.update(status='failed', error=str(error))
        report['real_cases'].append(info)
    stats = fit_normalization(samples) if samples else None
    # Development-only statistics in memory; not the formal training scaler/cache.
    report['source_sha256'] = {k: sha256(config[k]) for k in ('index', 'registry')}
    _, exposure_report = exposure_ids(config); report['exposure_manifest_check'] = exposure_report
    runtime_available = all(importlib.util.find_spec(m) for m in ('torch', 'torch_geometric'))
    if not runtime_available or not environment.get('cuda_available'):
        report['runtime_blocker'] = 'PyTorch/PyG unavailable or CUDA unavailable; no installation and no CPU training fallback'
        report.update(gpu_memory_peak_bytes=None, step_seconds=None, overnight_estimate=None)
    elif samples:
        import torch
        from losses import reconstruction_loss
        from train_reconstruction import build_model, collate_samples, device_for
        device = device_for(config); batch = collate_samples([normalize_sample(s, stats) for s in samples])
        batch = {k: v.to(device) for k, v in batch.items()}; torch.cuda.reset_peak_memory_stats()
        timings, parameters = {}, {}
        for kind in ('relation', 'set'):
            model = build_model(config, kind).to(device); parameters[kind] = model.parameter_counts()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']); elapsed = []
            for _ in range(2):
                if time.perf_counter()-started > 270: raise RuntimeError('Short-check budget reached before next optimizer step')
                torch.cuda.synchronize(); begin = time.perf_counter()
                pred, _ = model(batch); loss = reconstruction_loss(pred, batch['target'], batch['counts'], config['loss'])
                if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite short-check loss')
                optimizer.zero_grad(); loss.backward(); optimizer.step(); torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-begin); report['optimizer_steps'] += 1
            timings[kind] = elapsed
            del model, optimizer
        report.update(state='short_check_runnable_not_trained', gpu_memory_peak_bytes=torch.cuda.max_memory_allocated(),
                      step_seconds=timings, parameters=parameters,
                      overnight_estimate={'rough_seconds_two_models': sum(np.mean(t) for t in timings.values())*(4800/8)*50,
                        'assumption': '8-case check scaled to 4800 cases and 50 epochs; excludes IO/validation, batch32 memory unverified; not a promise'})
    report['elapsed_seconds'] = time.perf_counter()-started
    report['data_cases_passed'] = sum(r['status']=='data_adapter_passed' for r in report['real_cases'])
    write_json(ROOT/'results'/'environment.json', environment)
    write_json(ROOT/'results'/'verification_summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not rows or report['data_cases_passed'] != len(rows): raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default='config_v01.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker: run(load_config(args.config)); return
    # A parent timeout enforces the cumulative cap even if a CUDA call stalls.
    try:
        completed = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                                    '--config', str(Path(args.config).resolve()), '--worker'], timeout=300)
    except subprocess.TimeoutExpired:
        write_json(ROOT/'results'/'verification_summary.json', {'state': 'short_check_timeout_not_verified',
                   'time_limit_seconds': 300, 'formal_prepare_run': False, 'formal_training_run': False})
        raise SystemExit('Short check timed out; no automatic retry')
    raise SystemExit(completed.returncode)


if __name__ == '__main__': main()
