"""Public reader contract using tiny generated fixtures and Python's standard library."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class DatasetIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'results'/'cleanup'/'test_tmp').mkdir(parents=True,exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        path=ROOT/'results'/'cleanup'/'test_tmp'
        if not any(path.iterdir()): path.rmdir()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results'/'cleanup'/'test_tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);data=self.root/'data';data.mkdir()
        self.graph={'encoding_version':'0.3.2','sample':{'sample_id':'one'},
                    'graph':{'node_count':1,'edge_count':0,'block_geometry_wkt_metric':'POLYGON EMPTY'},
                    'nodes':[{'geometry_wkt_metric':'POLYGON EMPTY','height_m':8}],'edges':[]}
        with gzip.open(data/'one.json.gz','wt',encoding='utf-8') as f: json.dump(self.graph,f)
        self.rows=[{'sample_id':'one','case_status':'completed','graph_path':'one.json.gz'},
                   {'sample_id':'excluded','case_status':'excluded_no_valid_building','graph_path':''}]
        self.write_index()
        self.descriptor=self.root/'dataset.json'
        self.descriptor.write_text(json.dumps({'encoding_version':'0.3.2','data_root':'data',
                                                'index':'index.csv.gz'}),encoding='utf-8')

    def write_index(self):
        with gzip.open(self.root/'data'/'index.csv.gz','wt',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['sample_id','case_status','graph_path'])
            writer.writeheader();writer.writerows(self.rows)

    def reader(self):
        path=ROOT/'dataset_io.py'
        self.assertTrue(path.is_file(),'standalone dataset_io.py has not been implemented')
        spec=importlib.util.spec_from_file_location('standalone_graph_reader',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.GraphDataset(self.descriptor)

    def test_returns_original_json_without_current_directory_dependency(self):
        dataset=self.reader()
        self.assertEqual(dataset.get('one'),self.graph)

    def test_unknown_and_excluded_ids_never_fall_back(self):
        dataset=self.reader()
        with self.assertRaises(KeyError): dataset.get('unknown')
        with self.assertRaises(ValueError): dataset.get('excluded')

    def test_wrong_root_and_escape_paths_are_rejected(self):
        self.descriptor.write_text(json.dumps({'data_root':'missing','index':'index.csv.gz'}))
        with self.assertRaises(FileNotFoundError): self.reader()
        self.descriptor.write_text(json.dumps({'data_root':'data','index':'index.csv.gz'}))
        for path in ('../outside.json.gz','C:/outside.json.gz','/outside.json.gz'):
            self.rows[0]['graph_path']=path;self.write_index()
            with self.subTest(path):
                with self.assertRaises(ValueError): self.reader().get('one')

    def test_duplicate_ids_and_wrong_graph_identity_fail(self):
        self.rows.append(dict(self.rows[0]));self.write_index()
        with self.assertRaises(ValueError): self.reader().get('one')
        self.rows.pop();self.write_index()
        self.graph['sample']['sample_id']='another'
        with gzip.open(self.root/'data'/'one.json.gz','wt') as f: json.dump(self.graph,f)
        with self.assertRaises(ValueError): self.reader().get('one')


if __name__=='__main__': unittest.main()
