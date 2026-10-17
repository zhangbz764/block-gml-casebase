import importlib
import unittest
import numpy as np
from test_data_adapter import graph,config


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        try: self.e=importlib.import_module('evaluate_generation');self.a=importlib.import_module('data_adapter')
        except ImportError: self.fail('6D evaluation is not implemented')

    def test_duplicate_component_count_and_height_not_layout_diversity(self):
        v=np.zeros((8,5));v[:,3]=1.;v[:,4]=10
        h=v.copy();h[:,4]=20
        result=self.e.diversity([v,v.copy(),h],config())
        self.assertEqual(result['different_layouts'],1)
        self.assertEqual(len(result['pairs']),3)
        self.assertGreater(result['pairs'][1]['height_mae_m'],0)

    def test_all_failed_and_not_run_rows_are_kept(self):
        planned=[('c','R',1),('c','B',1)]
        rows=self.e.planned_rows(planned,{('c','R',1):{'outcome':'no_feasible','stop_reason':'timeout'}})
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1]['outcome'],'not_run')
        self.assertEqual(rows[0]['stop_reason'],'timeout')

    def test_readonly_page_keeps_failure_and_unrun_cards(self):
        from render_review import attempt_caption,page_html
        rows=[{'outcome':'no_feasible','stop_reason':'timeout'},{'outcome':'not_run'}]
        html=page_html([attempt_caption(r) for r in rows],'not accepted')
        self.assertIn('no_feasible',html);self.assertIn('timeout',html);self.assertIn('not_run',html)

if __name__=='__main__': unittest.main()
