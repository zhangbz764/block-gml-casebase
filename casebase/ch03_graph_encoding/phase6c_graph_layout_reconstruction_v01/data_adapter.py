"""Read-only v0.3.2 adaptation; IDs/WKT are trace/geometry, never model channels."""
from __future__ import annotations
import csv
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import numpy as np
from shapely import wkt
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent
BLOCK_FIELDS = tuple('elongation compactness bcr convexity rectangularity far bcd osr avh_w hcv footprint_cv avg_complexity avg_rectangularity avg_convexity orientation_entropy'.split())
NODE_FIELDS = tuple('footprint_compactness footprint_rectangularity footprint_convexity orientation_sin_2theta orientation_cos_2theta local_x_normalized local_y_normalized boundary_distance_normalized footprint_area_m2 height_m'.split())
EDGE_FIELDS = tuple('shared_boundary_ratio_min shared_boundary_ratio_mean footprint_gap_block_norm centroid_distance_block_norm orientation_similarity long_side_overlap_ratio long_side_offset_norm connection_axis_alignment_i connection_axis_alignment_j log_area_ratio_abs log_height_ratio_abs volume_ratio'.split())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_config(path='config_v01.json'):
    path = Path(path).resolve(); config = json.loads(path.read_text(encoding='utf-8'))
    for field in ('index', 'registry', 'graph_root'):
        config[field] = str((path.parent/config[field]).resolve())
    config['exposure_sources'] = [str(path.parent/p) for p in config['exposure_sources']]
    if not (0 < config['max_samples'] <= 4800 and 0 < config['max_per_city'] <= 200
            and 0 < config['max_reads_per_city'] <= 600 and config['min_nodes'] >= 8
            and config['max_nodes'] <= 64 and 0 < config['epochs'] <= 50):
        raise ValueError('Configuration exceeds first-run sample/node/epoch budget')
    return config


def true(value): return str(value).lower() in ('true', '1')


def value(source, field):
    if source.get('missing', {}).get(field) is True: return np.nan
    raw = source.get(field)
    if raw is None or isinstance(raw, bool): return np.nan
    try:
        result = float(raw)
        return result if math.isfinite(result) else np.nan
    except (ValueError, TypeError): return np.nan


def read_graph(row, graph_root):
    root = Path(graph_root).resolve(); raw = str(row.get('graph_path', ''))
    if not raw or Path(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ValueError('graph_path must be relative to Phase4B')
    path = (root/raw).resolve(); path.relative_to(root)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream: record = json.load(stream)
    if record.get('encoding_version') != '0.3.2' or not true(record.get('validation', {}).get('passed')):
        raise ValueError('Graph is not validated v0.3.2')
    for field in ('sample_id', 'city', 'block_id'):
        if str(record.get('sample', {}).get(field)) != str(row[field]):
            raise ValueError(f'Graph identity mismatch: {field}')
    if len(record['nodes']) != int(float(row['node_count'])): raise ValueError('Index/node count mismatch')
    return record


def block_geometry(record):
    geom = wkt.loads(record['graph']['block_geometry_wkt_metric'])
    if geom.geom_type not in ('Polygon', 'MultiPolygon') or geom.is_empty or not geom.is_valid:
        raise ValueError('Invalid block polygon')
    return geom


def coordinate_frame(record):
    x0, y0, x1, y1 = block_geometry(record).bounds
    L = max(x1-x0, y1-y0)
    if not math.isfinite(L) or L <= 0: raise ValueError('Invalid block coordinate scale')
    return {'center': [(x0+x1)/2, (y0+y1)/2], 'L': L}


def to_metric(xy, frame): return np.asarray(xy)*frame['L'] + np.asarray(frame['center'])


def make_targets(record):
    frame = coordinate_frame(record); targets = []
    ids = [str(node['building_id']) for node in record['nodes']]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate building IDs')
    for node in record['nodes']:
        geom = wkt.loads(node['geometry_wkt_metric']); height = value(node, 'height_m')
        area = value(node, 'footprint_area_m2')
        if geom.geom_type not in ('Polygon', 'MultiPolygon') or geom.is_empty or not geom.is_valid:
            raise ValueError('Missing or invalid building footprint')
        if geom.area <= 0 or not math.isfinite(area) or area <= 0 or not math.isfinite(height) or height <= 0:
            raise ValueError('Nonpositive or missing footprint/height')
        rect = geom.minimum_rotated_rectangle
        points = np.asarray(rect.exterior.coords)[:4]; vectors = np.roll(points, -1, axis=0)-points
        lengths = np.linalg.norm(vectors, axis=1); k = int(np.argmax(lengths))
        long, short = float(lengths[k]), float(lengths[(k+1)%4])
        if short <= 0: raise ValueError('Degenerate rectangle')
        theta = math.atan2(vectors[k, 1], vectors[k, 0]); center = np.array([rect.centroid.x, rect.centroid.y])
        xy = (center-np.asarray(frame['center']))/frame['L']
        targets.append([*xy, long/frame['L'], short/frame['L'], math.sin(2*theta), math.cos(2*theta), math.log1p(height)])
    target = np.asarray(targets, dtype=np.float64)
    if target.size == 0 or not np.isfinite(target).all(): raise ValueError('Invalid targets')
    return target, frame


def make_boundary_condition(record):
    geom = block_geometry(record); frame = coordinate_frame(record)
    positions = (np.arange(32)+.5)/32-.5
    return np.array([[geom.covers(Point(*to_metric([x, y], frame))) for x in positions]
                     for y in positions], dtype=np.float32)


def adapt_graph(record):
    target, frame = make_targets(record); nodes = record['nodes']; lookup = {str(n['building_id']): i for i, n in enumerate(nodes)}
    rows = []
    for node in nodes:
        row = [value(node, f) for f in NODE_FIELDS]
        # Node core xy formerly centroid/sqrt(block area). Use footprint centroid in bbox/L frame.
        geom = wkt.loads(node['geometry_wkt_metric'])
        row[5:7] = (np.asarray([geom.centroid.x, geom.centroid.y])-frame['center'])/frame['L']
        row[7] = value(node, 'boundary_distance_m')/frame['L']
        row[8] = value(node, 'footprint_area_m2')/frame['L']**2
        row[9] = math.log1p(value(node, 'height_m')); rows.append(row)
    attrs = record.get('block_attributes', {})
    source = {**attrs.get('values', {}), 'missing': attrs.get('missing', {})}
    edges, indices = [], []
    for edge in record.get('edges', []):
        i, j = lookup[str(edge['source'])], lookup[str(edge['target'])]
        if i == j: raise ValueError('Source Graph contains self edge')
        row = np.array([value(edge, f) for f in EDGE_FIELDS]); reverse = row.copy()
        reverse[7], reverse[8] = row[8], row[7]
        edges.extend([row, reverse]); indices.extend([[i, j], [j, i]])
    return {'node': np.asarray(rows), 'block': np.asarray([value(source, f) for f in BLOCK_FIELDS]),
            'edge': np.asarray(edges).reshape(-1, 12), 'edge_index': np.asarray(indices, dtype=np.int64).reshape(-1, 2).T,
            'target': target, 'boundary': make_boundary_condition(record), 'frame': frame,
            'node_ids': [str(n['building_id']) for n in nodes]}


def permute_sample(sample, permutation):
    permutation = np.asarray(permutation); inverse = np.argsort(permutation)
    if sorted(permutation.tolist()) != list(range(len(sample['node']))): raise ValueError('Not a node permutation')
    out = dict(sample); out['node'] = sample['node'][permutation]; out['target'] = sample['target'][permutation]
    out['edge_index'] = inverse[sample['edge_index']]
    out['node_ids'] = [sample['node_ids'][i] for i in permutation]
    return out


def fit_normalization(train_samples):
    if not train_samples: raise ValueError('Empty train split')
    stats = {}
    for key, width in [('node', 10), ('block', 15), ('edge', 12)]:
        data = np.concatenate([s[key].reshape(-1, width) for s in train_samples])
        observed = np.isfinite(data); counts = observed.sum(axis=0)
        mean = np.divide(np.where(observed, data, 0).sum(axis=0), counts, out=np.zeros(width), where=counts>0)
        variance = np.divide(np.where(observed, data-mean, 0).__pow__(2).sum(axis=0), counts, out=np.zeros(width), where=counts>0)
        stats[key] = {'mean': mean.tolist(), 'std': np.where(variance>1e-12, np.sqrt(variance), 1).tolist(),
                      'observed': counts.tolist()}
    heights = np.concatenate([s['target'][:, 6] for s in train_samples])
    stats['height'] = {'mean': float(heights.mean()), 'std': max(float(heights.std()), 1e-6)}
    return stats


def normalize_sample(sample, stats):
    out = dict(sample)
    for key in ('node', 'block', 'edge'):
        raw = sample[key]; missing = ~np.isfinite(raw); mean = np.asarray(stats[key]['mean']); std = np.asarray(stats[key]['std'])
        standardized = (np.where(missing, mean, raw)-mean)/std
        out[key] = np.concatenate([standardized, missing.astype(float)], axis=-1).astype(np.float32)
    out['target'] = sample['target'].copy().astype(np.float32)
    out['target'][:, 6] = (out['target'][:, 6]-stats['height']['mean'])/stats['height']['std']
    return out


def node_band(n): return '8-16' if n <= 16 else '17-32' if n <= 32 else '33-64'


def exposure_ids(config):
    ids, report = set(), []
    def walk(value):
        if isinstance(value, dict):
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
        elif isinstance(value, str) and value.startswith('P4B_'): ids.add(value)
    for pattern in config['exposure_sources']:
        paths = sorted(glob.glob(pattern))
        if not paths: report.append({'pattern': pattern, 'status': 'missing; unseen-test claim unavailable'})
        for filename in paths:
            path = Path(filename)
            try:
                if path.suffix == '.json': walk(json.loads(path.read_text(encoding='utf-8')))
                else:
                    with path.open(encoding='utf-8-sig', newline='') as f:
                        for row in csv.DictReader(f): walk(row)
                report.append({'path': str(path), 'sha256': sha256(path), 'status': 'read'})
            except (OSError, ValueError) as error: report.append({'path': str(path), 'status': str(error)})
    # Cases inspected by this stage's short check are also not unseen final-test evidence.
    short_check = ROOT/'results'/'verification_summary.json'
    if short_check.exists():
        walk(json.loads(short_check.read_text(encoding='utf-8')).get('real_cases', []))
        report.append({'path': str(short_check), 'sha256': sha256(short_check), 'status': 'read'})
    return ids, report


def assign_splits(rows, seed, cell_m=1000):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def merge(i, j): parent[find(j)] = find(i)
    keys = {}
    for i, row in enumerate(rows):
        city = row['city']
        labels = [('cell', city, math.floor(row['center_x']/cell_m), math.floor(row['center_y']/cell_m)),
                  ('block', city, row['block_id']), ('target', row['target_hash'])]
        if row.get('metric_target_hash'): labels.append(('metric_target', row['metric_target_hash']))
        labels += [('building', city, building) for building in row['building_ids']]
        for key in labels:
            if key in keys: merge(i, keys[key])
            else: keys[key] = i
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(find(i), []).append(row)
    for group in groups.values():
        group_id = stable_hash(sorted(r['sample_id'] for r in group))
        fraction = int(stable_hash([seed, group_id])[:16], 16)/16**16
        split = 'train' if fraction < .8 else 'val' if fraction < .9 else 'test'
        if split == 'test' and any(r['exposed'] for r in group): split = 'val'
        for row in group: row.update(group_id=group_id, split=split)
    return rows


def build_sample_manifest(config):
    exposed, exposure_report = exposure_ids(config); cities = {}
    with gzip.open(config['index'], 'rt', encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['case_status'] != 'completed' or not all(true(row.get(k)) for k in ('audit_passed', 'graph_readable', 'validation_passed')): continue
            n = int(float(row['node_count']))
            if config['min_nodes'] <= n <= config['max_nodes']:
                cities.setdefault(row['city'], {}).setdefault(node_band(n), []).append(row)
    samples, selected, rejected, city_counts = [], [], [], {}
    for city, bands in sorted(cities.items()):
        buckets = [sorted(bands.get(band, []), key=lambda r: stable_hash([config['seed'], r['sample_id']]))
                   for band in ('8-16', '17-32', '33-64')]
        candidates = [r for i in range(max(map(len, buckets), default=0)) for bucket in buckets for r in bucket[i:i+1]]
        reads = accepted = 0
        for row in candidates[:config['max_reads_per_city']]:
            if accepted >= config['max_per_city'] or len(samples) >= config['max_samples']: break
            reads += 1
            trace = {k: row[k] for k in ('sample_id', 'city', 'block_id', 'graph_path', 'node_count')}
            try:
                record = read_graph(row, config['graph_root']); sample = adapt_graph(record)
                # Meter-coordinate, order-independent target fingerprint; all cities share duplicate-target check.
                geometry_target = sample['target'].copy()
                geometry_target[:, :2] = to_metric(geometry_target[:, :2], sample['frame'])
                geometry_target[:, 2:4] *= sample['frame']['L']
                rounded = np.round(geometry_target, config['target_fingerprint_decimals']).tolist()
                rounded = [[0. if value == 0 else value for value in row] for row in rounded]
                normalized_rows = np.round(sample['target'], config['target_fingerprint_decimals']).tolist()
                normalized_rows = [[0. if value == 0 else value for value in row] for row in normalized_rows]
                sample.update(sample_id=row['sample_id'], city=city)
                selected.append({**trace, 'status': 'selected', 'reason': '', 'building_ids': sample['node_ids'],
                    'center_x': sample['frame']['center'][0], 'center_y': sample['frame']['center'][1],
                    'target_hash': stable_hash(sorted(normalized_rows)), 'metric_target_hash': stable_hash(sorted(rounded)),
                    'exposed': row['sample_id'] in exposed,
                    'source_sha256': sha256(Path(config['graph_root'])/row['graph_path'])})
                samples.append(sample); accepted += 1
            except (ValueError, KeyError, OSError, TypeError) as error:
                rejected.append({**trace, 'status': 'inapplicable', 'reason': str(error), 'split': ''})
        city_counts[city] = {'candidate_index_count': sum(map(len, buckets)), 'graphs_read': reads, 'selected': accepted}
    assign_splits(selected, config['seed'], config['spatial_cell_m'])
    for city, counts in city_counts.items():
        counts['splits'] = {split: sum(r['city']==city and r['split']==split for r in selected) for split in ('train', 'val', 'test')}
    for row, sample in zip(selected, samples, strict=True): sample['split'] = row['split']
    return selected+rejected, samples, {'cities': city_counts, 'exposure': exposure_report,
                                     'exposed_ids_count': len(exposed), 'split_rule': 'group hash 80/10/10; exposed test groups moved to val'}
