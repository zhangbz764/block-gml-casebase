"""Manual test-only evaluation after BOTH first runs complete; no test-guided tuning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from data_adapter import ROOT, block_geometry, node_band, read_graph, sha256, stable_hash, to_metric, write_json
from losses import match_buildings

METRICS = ('center_error_norm', 'center_error_m', 'long_relative_error', 'short_relative_error', 'height_mae_m',
           'pred_overlap_ratio', 'pred_outside_ratio', 'target_overlap_ratio', 'target_outside_ratio')


def restore_layout(target, frame, height_stats):
    layout = np.asarray(target, dtype=float).copy(); layout[:, :2] = to_metric(layout[:, :2], frame)
    layout[:, 2:4] *= frame['L']
    with np.errstate(over='ignore'): layout[:, 6] = np.expm1(layout[:, 6]*height_stats['std']+height_stats['mean'])
    if not np.isfinite(layout).all(): raise ValueError('Nonfinite inverse prediction')
    return layout


def rectangle_polygons(layout):
    rectangles = []
    for x, y, long, short, sin2, cos2, _ in layout:
        if min(long, short) <= 0: raise ValueError('Invalid rectangle sizes')
        angle = .5*np.arctan2(sin2, cos2); c, s = np.cos(angle), np.sin(angle)
        corners = np.array([[-long/2, -short/2], [long/2, -short/2], [long/2, short/2], [-long/2, short/2]])
        rectangles.append(Polygon(corners@np.array([[c, s], [-s, c]])+[x, y]))
    return rectangles


def geometry_diagnostics(layout, boundary):
    rectangles = rectangle_polygons(layout); total = sum(p.area for p in rectangles)
    overlap = (total-unary_union(rectangles).area)/total
    outside = sum(p.difference(boundary).area for p in rectangles)/total
    return float(max(0, overlap)), float(max(0, outside))


def case_metrics(pred, target, frame, boundary, height_stats, weights=None):
    i, j = match_buildings(pred, target, weights)
    pm = restore_layout(pred, frame, height_stats); tm = restore_layout(target, frame, height_stats)
    center = np.linalg.norm(np.asarray(pred)[i, :2]-np.asarray(target)[j, :2], axis=1).mean()
    sizes = np.abs(pm[i, 2:4]-tm[j, 2:4])/tm[j, 2:4]
    po, px = geometry_diagnostics(pm, boundary); to, tx = geometry_diagnostics(tm, boundary)
    return {'center_error_norm': float(center), 'center_error_m': float(center*frame['L']),
            'long_relative_error': float(sizes[:, 0].mean()), 'short_relative_error': float(sizes[:, 1].mean()),
            'height_mae_m': float(np.abs(pm[i, 6]-tm[j, 6]).mean()), 'pred_overlap_ratio': po,
            'pred_outside_ratio': px, 'target_overlap_ratio': to, 'target_outside_ratio': tx,
            'negative_height_ratio': float((pm[:, 6] < 0).mean())}


def select_review_cases(rows, seed):
    """6 seeded band-covered +3 A worst +3 signed differences, unique; seeded fill."""
    ordered = sorted(rows, key=lambda r: stable_hash([seed, r['sample_id']]))
    picked, seen = [], set()
    def add(row, reason):
        if row['sample_id'] in seen: return False
        picked.append({**row, 'selection_source': reason}); seen.add(row['sample_id']); return True
    bands = [[r for r in ordered if node_band(int(float(r['node_count']))) == b] for b in ('8-16', '17-32', '33-64')]
    for i in range(max(map(len, bands), default=0)):
        for band in bands:
            if len(picked) < 6 and i < len(band): add(band[i], 'seeded_band_coverage')
        if len(picked) >= 6: break
    worst = sorted(rows, key=lambda r: (-r['relation_center_error_norm'], r['sample_id']))
    count = 0
    for row in worst:
        if count >= 3: break
        count += add(row, 'A_position_error_largest')
    difference = lambda r: r['relation_center_error_norm']-r['set_center_error_norm']
    count = 0
    for sign, reason in [(-1, 'A_better_large_difference'), (1, 'B_better_large_difference')]:
        pool = sorted([r for r in rows if difference(r)*sign > 0], key=lambda r: -abs(difference(r)))
        for row in pool:
            if add(row, reason): count += 1; break
    for row in sorted(rows, key=lambda r: (-abs(difference(r)), r['sample_id'])):
        if count >= 3: break
        count += add(row, 'absolute_difference_largest')
    for row in ordered:
        if len(picked) >= min(12, len(rows)): break
        add(row, 'seeded_deduplicated_fill')
    return picked[:12]


def summarize(rows):
    groups = [('overall', 'all', rows)]
    groups += [('node_band', band, [r for r in rows if node_band(int(float(r['node_count'])))==band]) for band in ('8-16', '17-32', '33-64')]
    groups += [('city', city, [r for r in rows if r['city']==city]) for city in sorted({r['city'] for r in rows})]
    output = []
    for level, label, group in groups:
        if not group: continue
        for model in ('relation', 'set', 'A_minus_B'):
            for metric in METRICS:
                values = np.array([(r[f'relation_{metric}']-r[f'set_{metric}']) if model=='A_minus_B' else r[f'{model}_{metric}'] for r in group])
                q1, median, q3 = np.quantile(values, [.25, .5, .75])
                output.append(dict(group_type=level, group=label, model=model, metric=metric, blocks=len(group),
                                   mean=float(values.mean()), median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3-q1)))
    return output


def write_csv(path, rows):
    if not rows: raise ValueError('No rows to write')
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate(run_id, models, render_review):
    import torch
    from prepare_dataset import load_run
    from train_reconstruction import build_model, collate_samples, device_for, read_cache
    if models != 'relation,set': raise ValueError('Evaluate the paired relation,set models together')
    directory, lock = load_run(run_id); config = lock['config']; device = device_for(config)
    entries = json.loads((directory/'samples.json').read_text(encoding='utf-8'))
    test = [r for r in entries if r['split']=='test']
    if not test: raise ValueError('No test samples')
    height_stats = json.loads((directory/'normalization.json').read_text(encoding='utf-8'))['height']
    networks, hashes = {}, {}
    for kind in ('relation', 'set'):
        latest = torch.load(directory/kind/'latest.pt', map_location='cpu', weights_only=False)
        state = torch.load(directory/kind/'best.pt', map_location='cpu', weights_only=False)
        if not latest['completed']: raise ValueError(f'{kind} has not completed; test evaluation refused')
        for checkpoint in (state, latest):
            if checkpoint['model_kind'] != kind or checkpoint['config_hash'] != lock['config_hash'] or checkpoint['code_hash'] != lock['code_hash']:
                raise ValueError('Checkpoint identity mismatch')
        network = build_model(config, kind).to(device); network.load_state_dict(state['model']); network.eval()
        networks[kind] = network; hashes[kind] = sha256(directory/kind/'best.pt')
    previous = directory/'evaluation_manifest.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))['checkpoint_sha256'] != hashes:
        raise ValueError('Previously evaluated checkpoints changed; do not repeatedly use test to tune')
    rows, predictions = [], {}
    for entry in test:
        source = Path(config['graph_root'])/entry['graph_path']
        if sha256(source) != entry['source_sha256']: raise ValueError('Selected source Graph changed')
        record = read_graph(entry, config['graph_root']); boundary = block_geometry(record)
        sample = read_cache(directory, entry); batch = {k: v.to(device) for k, v in collate_samples([sample]).items()}
        row = {k: entry[k] for k in ('sample_id', 'city', 'node_count')}; per_model = {}
        for kind, network in networks.items():
            with torch.no_grad(): pred, _ = network(batch)
            pred = pred[0, :len(sample['node'])].cpu().numpy(); per_model[kind] = pred
            metrics = case_metrics(pred, sample['target'], entry['frame'], boundary, height_stats, config['loss'])
            row.update({f'{kind}_{key}': val for key, val in metrics.items()})
        row['A_minus_B_center_error_norm'] = row['relation_center_error_norm']-row['set_center_error_norm']
        rows.append(row); predictions[entry['sample_id']] = per_model
    np.savez_compressed(directory/'predictions.npz', **{f'{identifier}__{kind}': array for identifier, data in predictions.items() for kind, array in data.items()})
    write_csv(directory/'case_metrics.csv', rows); write_csv(ROOT/'results'/'metrics_summary.csv', summarize(rows))
    write_json(previous, {'state': 'evaluated_not_researcher_accepted', 'checkpoint_sha256': hashes,
               'test_count': len(test), 'config_hash': lock['config_hash'], 'code_hash': lock['code_hash']})
    if render_review:
        from visualize_reconstruction import render_review_page
        render_review_page(run_id, select_review_cases(rows, config['seed']), test, predictions, directory, config, height_stats)
    print(json.dumps({'state': 'evaluated_not_researcher_accepted', 'test_count': len(test)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-id', required=True)
    parser.add_argument('--models', default='relation,set'); parser.add_argument('--render-review', action='store_true')
    args = parser.parse_args(); evaluate(args.run_id, args.models, args.render_review)


if __name__ == '__main__': main()
