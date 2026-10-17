"""One assignment cost for training and evaluation. Torch is imported only for loss."""
import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_WEIGHTS = {'position': 1., 'size': 1., 'height': 1., 'direction': .1, 'smooth_l1_beta': 1.}


def matching_cost(prediction, target, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    p, t = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 7 or not len(p): raise ValueError('Expected equal nonempty Nx7 sets')
    if not np.isfinite(p).all() or not np.isfinite(t).all(): raise ValueError('Nonfinite assignment inputs')
    difference = np.abs(p[:, None, :]-t[None, :, :]); beta = weights['smooth_l1_beta']
    smooth = np.where(difference < beta, .5*difference**2/beta, difference-.5*beta)
    pu = p[:, 4:6]/np.maximum(np.linalg.norm(p[:, 4:6], axis=1, keepdims=True), 1e-8)
    tu = t[:, 4:6]/np.maximum(np.linalg.norm(t[:, 4:6], axis=1, keepdims=True), 1e-8)
    anisotropy = np.clip((t[:, 2]-t[:, 3])/np.maximum(t[:, 2], 1e-8), 0, 1)
    direction = (1-np.clip(pu@tu.T, -1, 1))*anisotropy[None, :]
    return (weights['position']*smooth[..., :2].mean(-1) + weights['size']*smooth[..., 2:4].mean(-1)
            + weights['height']*smooth[..., 6] + weights['direction']*direction)


def match_buildings(prediction, target, weights=None):
    return linear_sum_assignment(matching_cost(prediction, target, weights))


def reconstruction_loss(prediction, targets, counts, weights=None):
    import torch
    import torch.nn.functional as F
    weights = weights or DEFAULT_WEIGHTS; block_losses = []
    for pred, target, n in zip(prediction, targets, counts, strict=True):
        n = int(n); pred, target = pred[:n], target[:n]
        i, j = match_buildings(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), weights)
        p, t = pred[torch.as_tensor(i, device=pred.device)], target[torch.as_tensor(j, device=target.device)]
        smooth = F.smooth_l1_loss(p, t, reduction='none', beta=weights['smooth_l1_beta'])
        direction = 1-(F.normalize(p[:, 4:6], dim=-1)*F.normalize(t[:, 4:6], dim=-1)).sum(-1).clamp(-1, 1)
        anisotropy = ((t[:, 2]-t[:, 3])/t[:, 2].clamp_min(1e-8)).clamp(0, 1)
        block_losses.append((weights['position']*smooth[:, :2].mean(-1) + weights['size']*smooth[:, 2:4].mean(-1)
            + weights['height']*smooth[:, 6] + weights['direction']*direction*anisotropy).mean())
    return torch.stack(block_losses).mean()
