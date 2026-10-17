"""Runtime checks stay pending (not green) when torch/PyG are unavailable."""
import importlib.util
import inspect
import unittest
import numpy as np

AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ('torch', 'torch_geometric'))


@unittest.skipUnless(AVAILABLE, 'requires torch and torch_geometric; model runtime NOT verified')
class ModelTests(unittest.TestCase):
    def test_forward_backward_mask_and_permutation(self):
        import torch
        from models import ReconstructionModel
        from losses import reconstruction_loss
        from train_reconstruction import collate_samples
        from test_data_adapter import fixture
        from data_adapter import adapt_graph, fit_normalization, normalize_sample, permute_sample
        sample = adapt_graph(fixture())
        normalized = normalize_sample(sample, fit_normalization([sample]))
        extra = dict(normalized)
        extra['node'] = np.concatenate([normalized['node'], normalized['node'][:1]])
        extra['target'] = np.concatenate([normalized['target'], normalized['target'][:1]])
        extra['node_ids'] = normalized['node_ids']+['third']
        batch = collate_samples([normalized, extra])
        for kind in ('relation', 'set'):
            torch.manual_seed(20261009)
            model = ReconstructionModel(kind)
            pred, mask = model(batch)
            self.assertEqual(pred.shape, (2, 64, 7)); self.assertEqual(int(mask.sum()), 5)
            self.assertTrue(bool(torch.isfinite(pred).all())); self.assertTrue(bool((pred[..., 2:4] > 0).all()))
            loss = reconstruction_loss(pred, batch['target'], batch['counts'])
            shuffled = batch['target'].clone(); shuffled[:, :2] = shuffled[:, [1, 0]]
            self.assertAlmostEqual(float(loss.detach()), float(reconstruction_loss(pred, shuffled, batch['counts']).detach()), places=6)
            loss.backward(); self.assertTrue(any(p.grad is not None for p in model.parameters()))
            model.eval()
            other = collate_samples([permute_sample(normalized, [1, 0]), extra])
            torch.testing.assert_close(model(batch)[0], model(other)[0], atol=1e-5, rtol=1e-5)

    def test_decoder_interface_and_edge_sensitivity(self):
        import torch
        from models import LayoutDecoder, RelationEncoder, SetEncoder
        self.assertEqual(list(inspect.signature(LayoutDecoder.forward).parameters), ['self', 'z', 'boundary', 'counts'])
        self.assertNotIn('edge', str(inspect.signature(SetEncoder.forward)))
        torch.manual_seed(1)
        model = RelationEncoder().eval()
        x = torch.randn(3, 20); block = torch.randn(1, 30); group = torch.zeros(3, dtype=torch.long)
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        attrs = torch.randn(4, 24)
        a = model(x, block, group, edges, attrs)
        changed = attrs.clone(); changed[0, 0] += 30
        b = model(x, block, group, edges, changed)
        self.assertGreater(float((a-b).abs().max()), 1e-8)


if __name__ == '__main__': unittest.main()
