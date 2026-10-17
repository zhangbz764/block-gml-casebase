import importlib
import unittest
import math
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon


def config():
    return {'boundary_compression': .15, 'far_increase': .25, 'floor_height_m': 3.,
            'height_min_m': 3., 'rotation_bound_deg': 30., 'scale_bounds': [.85, 1.15],
            'gap_tolerance_absolute': .01, 'gap_tolerance_relative': .25,
            'relation_tolerance': .1, 'feasibility_weight': 1000., 'scale_weight': .01,
            'relation_weight': 1., 'outside_max': .0001, 'overlap_max': .0001,
            'far_relative_max': .02, 'height_tolerance_m': .000001,
            'duplicate_center_rms_max': .02, 'duplicate_scale_max': .05,
            'xtol': .0001, 'ftol': .0001, 'relation_check_tolerance': 1e-6}


def graph():
    footprints = [Polygon(box(10,10,20,20).exterior.coords, [box(12,12,14,14).exterior.coords])]
    footprints += [box(30+15*(i%3),10+25*(i//3),40+15*(i%3),20+25*(i//3)) for i in range(7)]
    return {'encoding_version': '0.3.2', 'validation': {'passed': True},
        'sample': {'sample_id': 'fixture', 'city': 'fixture_city', 'block_id': 'block'},
        'graph': {'block_geometry_wkt_metric': box(0,0,120,100).wkt},
        'nodes': [{'building_id': str(i), 'height_m': 10., 'orientation_deg': 0.,
                   'geometry_wkt_metric': p.wkt} for i,p in enumerate(footprints)],
        'edges': [{'source': '0', 'target': '1', 'footprint_gap_block_norm': 10/math.sqrt(12000),
                   'orientation_similarity': 1., 'long_side_overlap_ratio': 0., 'missing': {}}]}


class DataTests(unittest.TestCase):
    def setUp(self):
        try: self.a = importlib.import_module('data_adapter')
        except ImportError: self.fail('6D data_adapter is not implemented')

    def test_graph_path_is_relative_to_phase4b(self):
        from pathlib import Path
        root=Path('phase4b').resolve()
        self.assertEqual(self.a.graph_path('results/dataset/a.gz', root), root/'results/dataset/a.gz')
        for bad in ('../outside.json', 'C:/outside.json', ''):
            with self.assertRaises(ValueError): self.a.graph_path(bad, root)

    def test_isotropic_frame_and_hole_multipart_preserved(self):
        from layout_solver import make_layout
        g=graph(); g['nodes'][0]['geometry_wkt_metric']=MultiPolygon([Polygon(box(10,10,20,20).exterior.coords,
            [box(12,12,14,14).exterior.coords]),box(21,10,22,11)]).wkt
        g['edges'][0]['footprint_gap_block_norm']=8/math.sqrt(12000)
        case=self.a.case_from_graph(g,config()); v=self.a.reference_variables(case)
        original=case['geometries'][0]; layout=make_layout(case,v)
        self.assertEqual(layout['geometries'][0].geom_type,'MultiPolygon')
        self.assertAlmostEqual(layout['geometries'][0].area, original.area)
        v[0,3]=1.1; changed=make_layout(case,v)['geometries'][0]
        self.assertAlmostEqual(changed.area,original.area*1.21)
        self.assertEqual(len(changed.geoms[0].interiors),1)
        np.testing.assert_allclose(self.a.to_metric(v[:,:2],case['frame']),layout['centers'])

    def test_initials_are_deterministic_and_conditions_not_silently_changed(self):
        case=self.a.case_from_graph(graph(),config())
        np.testing.assert_equal(self.a.make_initial(case,20261013),self.a.make_initial(case,20261013))
        self.assertAlmostEqual(case['boundary'].area,12000*.85)
        self.assertAlmostEqual(case['target_far'],case['original_far']*1.25)

    def test_confirmation_is_required(self):
        with self.assertRaises(ValueError): self.a.require_confirmed({'state':'PREPARED_PENDING_CONFIRMATION'})

if __name__=='__main__': unittest.main()
