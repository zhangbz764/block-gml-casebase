"""Manual first run: CUDA required by default, no implicit CPU fallback or model resume."""
import argparse
import json
import time
import numpy as np
from data_adapter import ROOT, load_config, write_json
from prepare_dataset import load_run


def collate_samples(samples):
    import torch
    counts = [len(s['node']) for s in samples]; offsets = np.cumsum([0]+counts[:-1])
    targets = np.zeros((len(samples), 64, 7), dtype=np.float32)
    for i, sample in enumerate(samples): targets[i, :counts[i]] = sample['target']
    return {'node': torch.tensor(np.concatenate([s['node'] for s in samples]), dtype=torch.float32),
            'block': torch.tensor(np.stack([s['block'] for s in samples]), dtype=torch.float32),
            'edge': torch.tensor(np.concatenate([s['edge'] for s in samples]), dtype=torch.float32),
            'edge_index': torch.tensor(np.concatenate([s['edge_index']+offset for s, offset in zip(samples, offsets, strict=True)], axis=1), dtype=torch.long),
            'batch': torch.repeat_interleave(torch.arange(len(samples)), torch.tensor(counts)),
            'target': torch.tensor(targets), 'boundary': torch.tensor(np.stack([s['boundary'] for s in samples])),
            'counts': torch.tensor(counts, dtype=torch.long)}


def read_cache(directory, row):
    with np.load(directory/row['cache'], allow_pickle=False) as data: return {k: data[k] for k in data.files}


def device_for(config):
    import torch
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; stop. No automatic long CPU training.')
    if config['device'] not in ('cuda', 'cpu'): raise ValueError('Only explicit cuda/cpu supported')
    return torch.device(config['device'])


def seed_all(seed):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config, kind):
    from models import ReconstructionModel
    seed_all(config['seed'])
    return ReconstructionModel(kind, config['set_width'])


def train(config, run_id, kind, resume=False):
    import torch
    from torch.utils.data import DataLoader
    from losses import reconstruction_loss
    directory, lock = load_run(run_id, config); device = device_for(config)
    rows = json.loads((directory/'samples.json').read_text(encoding='utf-8'))
    train_rows = [r for r in rows if r['split']=='train']; val_rows = [r for r in rows if r['split']=='val']
    model_dir = directory/kind
    if model_dir.exists() and not resume: raise ValueError('Existing model run; use explicit --resume or preserve it')
    model_dir.mkdir(exist_ok=True)
    model = build_model(config, kind).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    start_epoch, best, stale, elapsed, history = 0, float('inf'), 0, 0., []
    if resume:
        state = torch.load(model_dir/'latest.pt', map_location=device, weights_only=False)
        if state['model_kind'] != kind or state['config_hash'] != lock['config_hash'] or state['code_hash'] != lock['code_hash']:
            raise ValueError('Checkpoint model/config/code mismatch')
        if state['completed']: raise ValueError('Run already completed; do not silently retrain')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_epoch, best, stale = state['epoch']+1, state['best'], state['stale']
        elapsed, history = state['elapsed_seconds'], state['history']
        torch.set_rng_state(state['rng_cpu'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([r.cpu() for r in state['rng_cuda']])
    begin = time.perf_counter()
    for epoch in range(start_epoch, config['epochs']):
        # Independent deterministic data order per epoch: restarting does not shift shuffle RNG.
        order = np.random.default_rng(config['seed']+epoch).permutation(len(train_rows))
        loaders = {'train': DataLoader([train_rows[i] for i in order], batch_size=config['batch_size'],
                       collate_fn=lambda r: collate_samples([read_cache(directory, x) for x in r])),
                   'val': DataLoader(val_rows, batch_size=config['batch_size'],
                       collate_fn=lambda r: collate_samples([read_cache(directory, x) for x in r]))}
        metrics = {}
        for split, loader in loaders.items():
            model.train(split == 'train'); total = nblocks = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.set_grad_enabled(split == 'train'):
                    pred, _ = model(batch); loss = reconstruction_loss(pred, batch['target'], batch['counts'], config['loss'])
                    if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite loss; stop without repairing geometry')
                    if split == 'train': optimizer.zero_grad(); loss.backward(); optimizer.step()
                count = len(batch['counts']); total += float(loss.detach())*count; nblocks += count
            metrics[split] = total/nblocks
        improved = metrics['val'] < best
        if improved: best, stale = metrics['val'], 0
        else: stale += 1
        completed = stale >= config['patience'] or epoch+1 == config['epochs']
        history.append({'epoch': epoch+1, **metrics})
        state = {'model_kind': kind, 'config_hash': lock['config_hash'], 'code_hash': lock['code_hash'],
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best': best,
                 'stale': stale, 'completed': completed, 'history': history,
                 'elapsed_seconds': elapsed+time.perf_counter()-begin, 'parameters': model.parameter_counts(),
                 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        # Latest + best only; temporary replacement file is immediately renamed.
        temp = model_dir/'latest.pt.tmp'; torch.save(state, temp); temp.replace(model_dir/'latest.pt')
        if improved:
            temp = model_dir/'best.pt.tmp'; torch.save(state, temp); temp.replace(model_dir/'best.pt')
        write_json(model_dir/'training_summary.json', {k: state[k] for k in ('model_kind', 'completed', 'epoch', 'best', 'elapsed_seconds', 'parameters', 'history')})
        print(json.dumps({'model': kind, 'epoch': epoch+1, **metrics}), flush=True)
        if completed: break


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default='config_v01.json'); parser.add_argument('--run-id', required=True)
    parser.add_argument('--model', choices=('relation', 'set'), required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try: train(load_config(args.config), args.run_id, args.model, args.resume)
    except RuntimeError as error:
        if 'out of memory' in str(error).lower(): raise SystemExit('GPU out of memory: stop; reduce batch only after updating run/config, no CPU fallback') from error
        raise


if __name__ == '__main__': main()
