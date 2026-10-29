import csv
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from casebase.graph_encoding.pipeline.final_screening import apply_final_rules,build_graph_manifest
from casebase.graph_encoding.pipeline.run_batches import (
    require_matching_building_count,summary_exit_code,append_csv_row,snapshot_csv_ledger,
    verify_frozen_manifest_files,sha256_file)
from casebase.graph_encoding.pipeline.verify_batches import completed_graph_problems

ROOT=Path(__file__).resolve().parents[1]


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): (ROOT/'results'/'cleanup'/'test_tmp').mkdir(parents=True,exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        path=ROOT/'results'/'cleanup'/'test_tmp'
        if not any(path.iterdir()): path.rmdir()

    def test_final_flags_and_upper_count_threshold_are_unchanged(self):
        rows=[]
        for name,flag,count in [('keep_s02','S02',10),('keep_s04','S04',10),
                                 ('exclude_s01','S01',10),('exclude_scale',None,369)]:
            row={'city':'Test','block_id':name,'lod1_building_count':count,'area_m2':400.,'bcr':.2,
                 'hard_flag_count':0,'hard_reasons':'','primary_hard_reason':''}
            row.update({f'H{i:02d}':False for i in range(1,6)})
            row.update({f'S{i:02d}':False for i in range(1,7)})
            if flag: row[flag]=True
            rows.append(row)
        result=apply_final_rules(pd.DataFrame(rows),369.)
        decisions=result.set_index('block_id')['final_decision'].to_dict()
        self.assertEqual(decisions,{'keep_s02':'eligible','keep_s04':'eligible',
            'exclude_s01':'exclude','exclude_scale':'exclude'})
        self.assertEqual(set(build_graph_manifest(result)['block_id']),{'keep_s02','keep_s04'})

    def test_count_mismatch_and_incomplete_runs_are_not_successful(self):
        with self.assertRaises(RuntimeError): require_matching_building_count(2,[{}])
        require_matching_building_count(1,[{}])
        self.assertNotEqual(summary_exit_code({'failed':1,'interrupted':0}),0)
        self.assertNotEqual(summary_exit_code({'failed':0,'interrupted':1}),0)
        self.assertEqual(summary_exit_code({'failed':0,'interrupted':0}),0)

    def test_append_ledger_snapshot_and_all_three_manifest_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results'/'cleanup'/'test_tmp') as d:
            root=Path(d);ledger=root/'ledger.csv';snapshot=root/'snapshot.csv'
            append_csv_row(ledger,{'id':'one'},['id']);append_csv_row(ledger,{'id':'two'},['id'])
            snapshot_csv_ledger(ledger,snapshot,['id'])
            with snapshot.open(encoding='utf-8-sig') as f:
                self.assertEqual([r['id'] for r in csv.DictReader(f)],['one','two'])
            source=root/'source';full=root/'full';part=root/'part'
            for file in (source,full,part): file.write_bytes(b'unchanged')
            hashes={'source_manifest_sha256':sha256_file(source),'full_run_manifest_v032.csv.gz':sha256_file(full),
                    'parts':{'part_01':sha256_file(part)}}
            verify_frozen_manifest_files(source,full,part,'part_01',hashes)
            part.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): verify_frozen_manifest_files(source,full,part,'part_01',hashes)

    def test_final_verification_checks_validation_and_metadata(self):
        record={'encoding_version':'0.3.2','validation':{'passed':False},'phase4b':{'part_id':'part_02'}}
        problems=completed_graph_problems(record,{'part_id':'part_01'})
        self.assertTrue(any('validation.passed' in x for x in problems))
        self.assertTrue(any('part_id' in x for x in problems))


if __name__=='__main__': unittest.main()
