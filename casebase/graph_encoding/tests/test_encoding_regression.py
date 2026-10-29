"""Frozen old-implementation outputs for tiny offline inputs; no SQL or real dataset."""
import json
from pathlib import Path
import unittest
from shapely import wkt
from shapely.geometry import box

from casebase.graph_encoding.core.encoding_v032 import encode_block_to_v032
from casebase.graph_encoding.core.export_v03 import apply_weak_edge_flags

FIXTURES=Path(__file__).parent/'fixtures'


def inputs(item):
    block={'block_id':'TEST','city':'Test City','country':'Testland','city_query':'test_city',
           'storage_srid':3857,'metric_srid':3857,'area_m2':400.,'bcr':.2,
           'avg_nn_distance':5.,'geometry':box(0,0,20,20)}
    buildings=[]
    for bid,text,height in item['buildings']:
        geometry=wkt.loads(text)
        buildings.append({'building_id':bid,'citygml_id':bid,'height':float(height),
            'area':geometry.area,'perimeter':geometry.length,'geometry':geometry})
    return {'sample_id':item['name'],'sample_groups':'offline_regression'},block,buildings


def stable(value):
    if isinstance(value,dict): return {k:stable(v) for k,v in value.items() if k!='runtime_seconds'}
    if isinstance(value,list): return [stable(x) for x in value]
    return value


class EncodingRegressionTests(unittest.TestCase):
    def test_same_nodes_cells_edges_geometry_and_attributes_as_old_encoder(self):
        examples=json.loads((FIXTURES/'encoding_inputs.json').read_text())
        expected=json.loads((FIXTURES/'encoding_expected.json').read_text())
        for item,want in zip(examples,expected):
            with self.subTest(item['name']):
                outcome=encode_block_to_v032(*inputs(item),sample_manifest_sha256='synthetic-membership')
                self.assertEqual(stable(vars(outcome)),want)
        self.assertEqual(expected[1]['record']['graph']['node_count'],1)
        self.assertEqual(expected[1]['record']['graph']['edge_count'],0)
        self.assertEqual(expected[2]['status'],'excluded_no_valid_building')

    def test_weak_flags_do_not_delete_or_reorder_edges(self):
        edges=[{'edge_id':str(i),'footprint_gap_block_norm':float(i if i<11 else 100),
                'shared_boundary_ratio_min':.001 if i==11 else 1.} for i in range(12)]
        flagged,summary=apply_weak_edge_flags(edges)
        self.assertEqual([e['edge_id'] for e in flagged],[str(i) for i in range(12)])
        self.assertTrue(flagged[-1]['combined_weak_candidate'])
        self.assertEqual(summary['edge_deletion_count'],0)
        self.assertNotIn('combined_weak_candidate',edges[-1])


if __name__=='__main__': unittest.main()
