"""Revision-only checks: synthetic inputs, no optimization or real Graph reads."""
from contextlib import ExitStack
import copy
import csv
import gzip
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import data_adapter as d
import prepare_experiment as p
from test_data_adapter import graph

CASE_IDS=['P4B_IT_BO_000304','P4B_US_BO_000046','P4B_DE_BE_000370']
COUNTS=[13,9,15]
SOURCE_ROOT=d.ROOT


class LevelsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='unit_levels_',dir=SOURCE_ROOT/'results'/'bulk')
        self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for module in (d,p): self.stack.enter_context(patch.object(module,'ROOT',self.root))
        self.stack.enter_context(patch('render_review.render_preview',return_value=None)) # only slow picture output is substituted
        (self.root/'experiment_contract_v01.md').write_text('synthetic contract',encoding='utf-8')
        registry=self.root/'registry.tsv';registry.write_text('field\trole\n'+''.join(f'edge.{f}\tcore_model\n' for f in d.RELATIONS),encoding='utf-8')
        rows=[]
        for identifier,n in zip(CASE_IDS,COUNTS):
            g=graph();g['sample'].update(sample_id=identifier,block_id=identifier)
            while len(g['nodes'])<n:
                node=copy.deepcopy(g['nodes'][2]);node['building_id']=str(len(g['nodes']));g['nodes'].append(node)
            filename=identifier+'.json';d.write_json(self.root/filename,g)
            rows.append(dict(sample_id=identifier,block_id=identifier,city='fixture_city',node_count=n,
                             case_status='completed',audit_passed='True',graph_path=filename))
        self.index=self.root/'index.csv.gz'
        with gzip.open(self.index,'wt',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        self.registry=registry

    def level_config(self,level):
        path=SOURCE_ROOT/f'config_{level}_v02.json'
        self.assertTrue(path.exists(),'three v0.2 configurations are missing')
        cfg=d.load_config(path);cfg.update(index=str(self.index),registry=str(self.registry),graph_root=str(self.root))
        return cfg

    def test_configs_make_correct_combined_pressure_and_distinct_initials(self):
        cases=[]
        for level,compression,increase,floor_change in [('light',.05,.10,.045),('medium',.15,.25,.0625),('strong',.30,.60,.12)]:
            cfg=self.level_config(level);self.assertEqual(cfg['selected_case_ids'],CASE_IDS)
            self.assertEqual(cfg['boundary_compression'],compression);self.assertEqual(cfg['far_increase'],increase)
            self.assertEqual(cfg['seeds'],list(range(20261013,20261018)))
            self.assertEqual(cfg['max_evaluations'],6000);self.assertEqual(cfg['max_seconds'],600)
            case=d.case_from_graph(d.read_json(self.root/(CASE_IDS[0]+'.json')),cfg);cases.append(case)
            ratio=case['target_far']*case['boundary'].area/(case['original_far']*case['original_boundary'].area)-1
            self.assertAlmostEqual(ratio,floor_change)
            np.testing.assert_equal(d.make_initial(case,20261013),d.make_initial(case,20261013))
        self.assertEqual({c['height_cap'] for c in cases},{10.})
        self.assertFalse(np.allclose(d.to_metric(d.make_initial(cases[0],20261013)[:,:2],cases[0]['frame']),
                                    d.to_metric(d.make_initial(cases[2],20261013)[:,:2],cases[2]['frame'])))

    def test_records_freeze_isolated_old_results_preserved_and_confirm_independent(self):
        root_records=self.root/'results';root_records.mkdir(exist_ok=True)
        for name in ('manifest.json','freeze.json','environment.json'):
            (root_records/name).write_text('OLD ROOT RECORD',encoding='utf-8')
        old=d.run_path('adapt_v01');d.write_json(old/'manifest.json',{'historical':True})
        before={str(f):f.read_bytes() for f in self.root.rglob('*') if f.is_file()}
        cfg1=self.level_config('light');cfg2=self.level_config('medium')
        p.prepare_cases(cfg1,'adapt_v02_light');p.prepare_cases(cfg2,'adapt_v02_medium')
        self.assertTrue(hasattr(d,'record_path'),'per-run compact record path missing')
        self.assertNotEqual(d.record_path('adapt_v02_light'),d.record_path('adapt_v02_medium'))
        p.confirm('adapt_v02_light',notes={s:'synthetic organization' for s in CASE_IDS})
        self.assertTrue((d.record_path('adapt_v02_light')/'freeze.json').exists())
        self.assertFalse((d.record_path('adapt_v02_medium')/'freeze.json').exists())
        self.assertEqual(d.read_json(d.record_path('adapt_v02_medium')/'manifest.json')['state'],'PREPARED_PENDING_CONFIRMATION')
        from evaluate_generation import evaluate_generation
        evaluation=evaluate_generation('adapt_v02_light',False) # no saved layouts, no objective or solve calls
        self.assertEqual(evaluation['counts']['planned'],30)
        self.assertEqual(evaluation['counts']['not_run'],30)
        self.assertTrue((d.record_path('adapt_v02_light')/'metrics_summary.csv').exists())
        self.assertTrue((d.record_path('adapt_v02_light')/'phase6d_evidence_pack.md').exists())
        for filename,content in before.items(): self.assertEqual(Path(filename).read_bytes(),content)
        changed=copy.deepcopy(cfg1);changed['far_increase']=.12
        with self.assertRaises(ValueError): p.prepare_cases(changed,'adapt_v02_light')
        with self.assertRaises(ValueError): p.confirm('adapt_v02_medium',notes={s:'different organization' for s in CASE_IDS})

    def test_cross_level_missing_partial_failure_and_case_identity(self):
        spec=importlib.util.find_spec('summarize_levels')
        self.assertIsNotNone(spec,'cross-level summary not implemented')
        s=importlib.import_module('summarize_levels');self.stack.enter_context(patch.object(s,'ROOT',self.root))
        for level in ('light','medium'):
            cfg=self.level_config(level);p.prepare_cases(cfg,'adapt_v02_'+level)
        cfg=self.level_config('strong');d.write_json(self.root/'config_strong_v02.json',cfg)
        rows=[{'case_id':CASE_IDS[0],'mode':'R','seed':20261013,'outcome':'no_feasible','stop_reason':'timeout','feasible':False,
               'relation_loss':.4,'center_change_rms_L':.2}]
        d.write_json(d.record_path('adapt_v02_light')/'evaluation_summary.json',{'run_id':'adapt_v02_light','level_id':'light','all_rows':rows,
            'case_summaries':[{'case_id':CASE_IDS[0],'mode':'R','feasible':0,'nonduplicate_feasible':0}]})
        result=s.summarize_levels(['adapt_v02_light','adapt_v02_medium','adapt_v02_strong'])
        self.assertEqual(result['counts']['planned'],90);self.assertEqual(result['counts']['not_run'],89)
        keyed={(r['case_id'],r['level_id'],r['mode']):r for r in result['rows']}
        self.assertEqual(keyed[(CASE_IDS[0],'light','R')]['failed_count'],1)
        self.assertEqual(keyed[(CASE_IDS[0],'light','R')]['timeout_count'],1)
        self.assertEqual(keyed[(CASE_IDS[0],'light','R')]['status'],'部分完成')
        self.assertIsNone(keyed[(CASE_IDS[0],'medium','R')]['feasible_count'])
        self.assertIsNone(keyed[(CASE_IDS[0],'strong','R')]['nonduplicate_feasible'])
        self.assertEqual(keyed[(CASE_IDS[0],'strong','R')]['status'],'未运行')
        self.assertEqual(len(keyed),18)
        s.write_outputs(result,True)
        with (self.root/'results/levels_summary.csv').open(encoding='utf-8-sig') as stream:
            saved=list(csv.DictReader(stream))
        self.assertEqual(len(saved),18)
        missing=next(r for r in saved if (r['case_id'],r['level_id'],r['mode'])==(CASE_IDS[0],'strong','R'))
        self.assertEqual(missing['feasible_count'],'')
        self.assertIn('未测',(self.root/'results/review/levels_v02/levels_overview.html').read_text(encoding='utf-8'))


if __name__=='__main__': unittest.main()
