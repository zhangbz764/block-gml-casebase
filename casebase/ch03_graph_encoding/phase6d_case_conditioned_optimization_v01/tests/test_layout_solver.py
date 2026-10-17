import importlib
import unittest
import numpy as np
from test_data_adapter import graph,config


class SolverTests(unittest.TestCase):
    def setUp(self):
        try: self.s=importlib.import_module('layout_solver');self.a=importlib.import_module('data_adapter')
        except ImportError: self.fail('6D solver is not implemented')
        self.case=self.a.case_from_graph(graph(),config());self.v=self.a.reference_variables(self.case)

    def test_common_terms_identical_and_zero_relation_weight(self):
        layout=self.s.make_layout(self.case,self.v)
        r=self.s.evaluate_layout(self.case,layout,'R');b=self.s.evaluate_layout(self.case,layout,'B')
        self.assertEqual(r['common_loss'],b['common_loss'])
        self.case['config']['relation_weight']=0
        self.assertEqual(self.s.evaluate_layout(self.case,layout,'R')['objective'],b['objective'])

    def test_geometric_violations_and_equivalent_far(self):
        initial=self.s.make_layout(self.case,self.v)
        result=self.s.evaluate_layout(self.case,initial,'B')
        self.assertAlmostEqual(result['overlap_ratio'],0)
        expected=sum(g.area*h/3 for g,h in zip(initial['geometries'],initial['heights']))/self.case['boundary'].area
        self.assertAlmostEqual(result['equivalent_far'],expected)
        moved=self.v.copy();moved[0,:2]=[5,5]
        self.assertGreater(self.s.evaluate_layout(self.case,self.s.make_layout(self.case,moved),'B')['outside_ratio'],0)

    def test_relation_original_recompute_missing_and_symmetric(self):
        self.assertLessEqual(max(self.case['relation_check'].values()),1e-6)
        from shapely.geometry import box
        a,b=box(0,0,10,2),box(2,4,12,6)
        self.assertAlmostEqual(self.s.pair_relations(a,b,0,0,100)['long_side_overlap_ratio'],.8)
        self.assertEqual(self.s.pair_relations(a,b,0,0,100),self.s.pair_relations(b,a,0,0,100))
        self.case['edges'][0]['reference']['orientation_similarity']=None
        result=self.s.evaluate_layout(self.case,self.s.make_layout(self.case,self.v),'R')
        self.assertEqual(result['relation_counts']['orientation_similarity']['skipped'],1)

    def test_budget_preserves_candidate_and_records_stop(self):
        result=self.s.solve_layout(self.case,self.v,'R',{'max_evaluations':1,'max_seconds':10})
        self.assertLessEqual(result['evaluations'],1)
        self.assertIsNotNone(result['layout'])
        self.assertEqual(result['stop_reason'],'evaluation_limit')
        timed=self.s.solve_layout(self.case,self.v,'B',{'max_evaluations':10,'max_seconds':0})
        self.assertEqual(timed['stop_reason'],'timeout');self.assertEqual(timed['evaluations'],0)

if __name__=='__main__': unittest.main()
