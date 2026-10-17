"""Formal preparation is MANUAL. No Graph/DB writes. Freeze before train-only fitting."""
import argparse
import csv
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from data_adapter import (ROOT, BLOCK_FIELDS, NODE_FIELDS, EDGE_FIELDS, build_sample_manifest,
                          fit_normalization, normalize_sample, load_config, sha256, stable_hash, write_json)


def environment_info():
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': {}}
    for name in ('numpy', 'scipy', 'shapely', 'torch', 'torch-geometric', 'matplotlib'):
        try: result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name] = None
    if result['packages']['torch']:
        try:
            import torch
            result.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda)
            if torch.cuda.is_available():
                result.update(gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        except (ImportError, OSError, RuntimeError) as error: result['torch_error'] = str(error)
    utility = shutil.which('nvidia-smi')
    if utility:
        try:
            probe = subprocess.run([utility, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=5, check=True)
            result['gpu_hardware_probe'] = probe.stdout.strip()
            result['gpu_hardware_probe_is_not_cuda_runtime_verification'] = True
        except (OSError, subprocess.SubprocessError) as error: result['gpu_probe_error'] = str(error)
    return result


def code_hash():
    files = sorted(ROOT.glob('*.py'))+[ROOT/'experiment_contract_v01.md']
    return stable_hash({p.name: sha256(p) for p in files})


def run_directory(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id): raise ValueError('Unsafe run-id')
    return ROOT/'results'/'bulk'/run_id


def source_identity(config):
    return {key: sha256(config[key]) for key in ('index', 'registry')}


def write_manifest(path, rows):
    fields = ['sample_id', 'city', 'block_id', 'node_count', 'graph_path', 'status', 'reason', 'split', 'group_id',
              'exposed', 'center_x', 'center_y', 'target_hash', 'metric_target_hash', 'source_sha256', 'building_ids']
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            output = {key: row.get(key, '') for key in fields}
            output['building_ids'] = json.dumps(output['building_ids'], ensure_ascii=False)
            writer.writerow(output)


def check_registry(config):
    with Path(config['registry']).open(encoding='utf-8-sig', newline='') as stream: rows = list(csv.DictReader(stream, delimiter='\t'))
    actual = {r['field'].split('.')[-1] for r in rows if r['role'] == 'core_model'}
    required = set(BLOCK_FIELDS+NODE_FIELDS[:8]+EDGE_FIELDS)
    if not required <= actual: raise ValueError(f'Core registry mismatch: {sorted(required-actual)}')


def load_run(run_id, config=None):
    directory = run_directory(run_id); lock = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if config is not None and stable_hash(config) != lock['config_hash']: raise ValueError('run-id configuration mismatch')
    if code_hash() != lock['code_hash']: raise ValueError('Code/contract changed since preparation; refuse mixed-version run')
    freeze = json.loads((directory/'freeze.json').read_text(encoding='utf-8'))
    for name, expected in freeze['output_sha256'].items():
        if sha256(directory/name) != expected: raise ValueError(f'Frozen asset changed: {name}')
    return directory, lock


def prepare(config, run_id):
    directory = run_directory(run_id); check_registry(config)
    if directory.exists():
        _, existing = load_run(run_id, config)
        if existing['source_sha256'] != source_identity(config): raise ValueError('Frozen source index/registry changed')
        print('Already prepared with identical configuration; preserved'); return
    compact = ROOT/'results'; compact.mkdir(exist_ok=True)
    if (compact/'manifest.json').exists():
        old = json.loads((compact/'manifest.json').read_text(encoding='utf-8'))
        if old.get('run_id') != run_id: raise ValueError('Compact results belong to another run; preserve them before starting a new run')
    rows, samples, report = build_sample_manifest(config)
    counts = Counter(s['split'] for s in samples)
    if any(counts[k] == 0 for k in ('train', 'val', 'test')):
        raise ValueError(f'Empty split after grouping: {dict(counts)}; review before training')
    directory.mkdir(parents=True)
    # Freeze selection, split and contract BEFORE statistics are fitted.
    write_manifest(directory/'sample_manifest.csv', rows)
    lock = {'run_id': run_id, 'config': config, 'config_hash': stable_hash(config), 'code_hash': code_hash(),
            'source_sha256': source_identity(config), 'counts': dict(counts), 'selected': len(samples),
            'inapplicable': sum(r['status']=='inapplicable' for r in rows), 'selection_report': report,
            'sample_manifest_sha256': sha256(directory/'sample_manifest.csv'),
            'normalization': 'bulk/{run_id}/normalization.json; train split only'}
    write_json(directory/'manifest.json', lock)
    stats = fit_normalization([s for s in samples if s['split']=='train'])
    write_json(directory/'normalization.json', stats)
    metadata = []
    cache = directory/'cache'; cache.mkdir()
    for i, sample in enumerate(samples):
        normalized = normalize_sample(sample, stats)
        filename = f'{i:05d}.npz'
        np.savez_compressed(cache/filename, **{k: normalized[k] for k in ('node', 'block', 'edge', 'edge_index', 'target', 'boundary')})
        row = next(r for r in rows if r['sample_id']==sample['sample_id'])
        metadata.append({k: row[k] for k in ('sample_id', 'city', 'block_id', 'graph_path', 'source_sha256', 'split', 'group_id', 'node_count')} |
                        {'cache': f'cache/{filename}', 'frame': sample['frame']})
    write_json(directory/'samples.json', metadata)
    names = ['manifest.json', 'sample_manifest.csv', 'normalization.json', 'samples.json']+[f'cache/{p.name}' for p in sorted(cache.glob('*.npz'))]
    freeze = {'state': 'prepared_not_trained', 'run_id': run_id, 'output_sha256': {name: sha256(directory/name) for name in names}}
    write_json(directory/'freeze.json', freeze)
    # Small, inspectable tracked records; no copied source Graphs.
    for name in ('manifest.json', 'freeze.json', 'sample_manifest.csv'):
        (compact/name).write_bytes((directory/name).read_bytes())
    write_json(compact/'environment.json', environment_info())
    print(json.dumps({'run_id': run_id, 'counts': dict(counts), 'selected': len(samples)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default='config_v01.json'); parser.add_argument('--run-id', required=True)
    args = parser.parse_args(); prepare(load_config(args.config), args.run_id)


if __name__ == '__main__': main()
