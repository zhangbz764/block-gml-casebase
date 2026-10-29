"""Read original compressed graph JSON using only Python's standard library.

The descriptor is the only path anchor; importing this file never touches SQL,
Torch, geometry libraries, legacy phases or the dataset itself.
"""
import csv
import gzip
import json
from pathlib import Path,PurePosixPath,PureWindowsPath


def safe_relative(value):
    """Reject absolute, Windows-drive and parent-traversing paths on every platform."""
    if not isinstance(value,str) or not value or '\x00' in value:
        raise ValueError('Empty or invalid dataset-relative path')
    posix=PurePosixPath(value.replace('\\','/'))
    if posix.is_absolute() or PureWindowsPath(value).drive or '..' in posix.parts:
        raise ValueError('Path must stay inside its dataset root: '+value)
    return Path(*posix.parts)


def inside(root,value):
    result=(root/safe_relative(value)).resolve()
    if not result.is_relative_to(root): raise ValueError('Path escapes dataset root: '+value)
    return result


class GraphDataset:
    def __init__(self,descriptor_path):
        self.descriptor_path=Path(descriptor_path).resolve()
        self.descriptor=json.loads(self.descriptor_path.read_text(encoding='utf-8-sig'))
        self.data_root=inside(self.descriptor_path.parent,self.descriptor['data_root'])
        if not self.data_root.is_dir(): raise FileNotFoundError('Dataset data_root does not exist: '+str(self.data_root))
        self.index_path=inside(self.data_root,self.descriptor['index'])
        if not self.index_path.is_file(): raise FileNotFoundError('Dataset index does not exist: '+str(self.index_path))
        self._index=None

    def _load_index(self):
        # Cache only lookup fields, not all attributes or any graph contents.
        index={}
        opener=gzip.open if self.index_path.suffix=='.gz' else open
        with opener(self.index_path,'rt',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            required={'sample_id','graph_path'}
            if not required.issubset(reader.fieldnames or []): raise ValueError('Index is missing required fields')
            status_field=self.descriptor.get('status_field',
                'case_status' if 'case_status' in reader.fieldnames else 'status')
            if status_field not in reader.fieldnames: raise ValueError('Index is missing its status field')
            for row in reader:
                sid=row['sample_id'];path=row['graph_path']
                if not sid or sid in index: raise ValueError('Empty or duplicate sample_id: '+sid)
                if path: safe_relative(path)
                if row[status_field]=='completed' and not path: raise ValueError('Completed index entry has no graph: '+sid)
                index[sid]=(row[status_field],path)
        self._index=index

    def get(self,sample_id):
        if self._index is None: self._load_index()
        if sample_id not in self._index: raise KeyError('Unknown sample_id: '+str(sample_id))
        status,path=self._index[sample_id]
        if status!='completed': raise ValueError('Graph unavailable for '+str(sample_id)+': status='+status)
        filename=inside(self.data_root,path)
        opener=gzip.open if filename.suffix=='.gz' else open
        with opener(filename,'rt',encoding='utf-8') as f: record=json.load(f)
        if record.get('sample',{}).get('sample_id')!=sample_id:
            raise ValueError('Graph identity does not match index: '+str(sample_id))
        version=self.descriptor.get('encoding_version')
        if version and record.get('encoding_version')!=version:
            raise ValueError('Graph encoding_version does not match descriptor')
        return record
