import unittest
import numpy as np
from shapely.geometry import box
from losses import match_buildings, matching_cost
from evaluate_reconstruction import case_metrics, select_review_cases


class EvaluationTests(unittest.TestCase):
    def test_unordered_exact_reconstruction(self):
        target = np.array([[-.2, 0, .1, .05, 0, 1, 2.], [.2, 0, .1, .05, 0, 1, 2.]])
        pred = target[::-1].copy()
        i, j = match_buildings(pred, target)
        self.assertEqual(float(matching_cost(pred, target)[i, j].sum()), 0)
        metrics = case_metrics(pred, target, {'center': [50, 50], 'L': 100}, box(0, 0, 100, 100),
                               {'mean': 0., 'std': 1.})
        self.assertEqual(metrics['center_error_m'], 0)
        self.assertEqual(metrics['height_mae_m'], 0)
        self.assertEqual(metrics['pred_outside_ratio'], 0)

    def test_target_overlap_reference_not_hidden(self):
        target = np.array([[0, 0, .2, .1, 0, 1, 2.], [0, 0, .2, .1, 0, 1, 2.]])
        metrics = case_metrics(target, target, {'center': [0, 0], 'L': 100}, box(-2, -2, 2, 2),
                               {'mean': 0., 'std': 1.})
        self.assertAlmostEqual(metrics['target_overlap_ratio'], .5)
        self.assertGreater(metrics['target_outside_ratio'], .5)

    def test_review_deduplicates_and_records_selection(self):
        rows = [dict(sample_id=str(i), node_count=8+i*3, relation_center_error_norm=i/10,
                     set_center_error_norm=(12-i)/10) for i in range(20)]
        selected = select_review_cases(rows, 20261009)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({r['sample_id'] for r in selected}), 12)
        self.assertTrue(all(r['selection_source'] for r in selected))


if __name__ == '__main__': unittest.main()
