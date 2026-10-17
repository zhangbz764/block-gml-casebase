"""Catch frame mixing, hole filling, leakage and endpoint permutation errors."""
import unittest
import numpy as np
from shapely.geometry import Polygon, box
import data_adapter as adapter


def fixture():
    nodes = []
    for i, x in enumerate((20, 60)):
        nodes.append({'building_id': str(i), 'geometry_wkt_metric': box(x, 40, x+10, 45).wkt,
                      'height_m': 9., 'footprint_area_m2': 50., 'boundary_distance_m': 5.,
                      'missing': {}, 'representative_point_x': x+5, 'representative_point_y': 42.5})
    return {'graph': {'block_geometry_wkt_metric': box(10, 20, 110, 70).wkt}, 'nodes': nodes,
            'block_attributes': {'values': {}, 'missing': {}},
            'edges': [{'source': '0', 'target': '1', 'connection_axis_alignment_i': .2,
                       'connection_axis_alignment_j': None, 'missing': {}}]}


class AdapterTests(unittest.TestCase):
    def test_isotropic_frame_and_rectangle_target(self):
        target, frame = adapter.make_targets(fixture())
        np.testing.assert_allclose(frame['center'], [60, 45])
        self.assertEqual(frame['L'], 100)
        np.testing.assert_allclose(target[0, :4], [-.35, -.025, .1, .05])
        np.testing.assert_allclose(adapter.to_metric(target[:, :2], frame), [[25, 42.5], [65, 42.5]])
        self.assertTrue(np.isfinite(target).all())

    def test_holes_stay_empty(self):
        polygon = Polygon(box(0, 0, 100, 100).exterior.coords, [box(25, 25, 75, 75).exterior.coords])
        record = fixture(); record['graph']['block_geometry_wkt_metric'] = polygon.wkt
        mask = adapter.make_boundary_condition(record)
        self.assertEqual(mask.shape, (32, 32)); self.assertEqual(mask[16, 16], 0)
        self.assertEqual(mask[2, 2], 1)

    def test_permutation_updates_edges_and_targets(self):
        sample = adapter.adapt_graph(fixture())
        permuted = adapter.permute_sample(sample, [1, 0])
        np.testing.assert_equal(permuted['target'], sample['target'][[1, 0]])
        np.testing.assert_equal(permuted['edge_index'], [[1, 0], [0, 1]])
        self.assertEqual(sample['edge'][0, 7], .2)
        self.assertTrue(np.isnan(sample['edge'][1, 7]))
        self.assertEqual(sample['edge'][1, 8], .2)

    def test_training_only_scaler(self):
        a = adapter.adapt_graph(fixture()); b = adapter.adapt_graph(fixture())
        b['node'][:, 0] = 999
        stats = adapter.fit_normalization([a])
        transformed = adapter.normalize_sample(b, stats)
        self.assertEqual(stats['node']['mean'][0], 0)
        self.assertTrue(np.isfinite(transformed['node']).all())
        self.assertEqual(transformed['node'].shape[1], 20)

    def test_shared_buildings_targets_cells_and_exposure_do_not_leak(self):
        rows = [dict(sample_id=str(i), city='c', block_id=str(i), center_x=x, center_y=0,
                     building_ids=ids, target_hash=h, exposed=exposed)
                for i, x, ids, h, exposed in [(0, 0, ['a'], 'h0', False),
                    (1, 5000, ['a'], 'h1', True), (2, 9000, ['b'], 'h1', False),
                    (3, 9200, ['c'], 'h3', False)]]
        adapter.assign_splits(rows, 20261009)
        self.assertEqual(len({r['group_id'] for r in rows}), 1)
        self.assertNotIn('test', {r['split'] for r in rows})

    def test_invalid_height_rejected(self):
        record = fixture(); record['nodes'][0]['height_m'] = 0
        with self.assertRaises(ValueError): adapter.make_targets(record)


if __name__ == '__main__': unittest.main()
