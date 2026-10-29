"""Freeze this refactored implementation for a new manual run; never overwrite historical hashes."""
import argparse
import json
import re
from pathlib import Path
from .code_integrity import ROOT,CRITICAL_CODE_FILES,compute_code_manifest
from .run_batches import (SPEC,SOURCE_MANIFEST_PATH,FULL_MANIFEST_PATH,MANIFEST_HASHES_PATH,
                          verify_frozen_manifest_files)

def prepare_run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id): raise ValueError('unsafe run id')
    output=ROOT/'results'/'runs'/run_id
    if output.exists(): raise FileExistsError('run already exists: '+run_id)
    hashes=json.loads(MANIFEST_HASHES_PATH.read_text(encoding='utf-8'))
    if hashes['source_manifest_sha256']!=SPEC['source_manifest_sha256']:
        raise ValueError('origin manifest/spec mismatch')
    for part_id in hashes['parts']:
        verify_frozen_manifest_files(SOURCE_MANIFEST_PATH,FULL_MANIFEST_PATH,
            ROOT/'results/metadata/encoding/parts'/f'{part_id}_manifest.csv.gz',part_id,hashes)
    bundle=compute_code_manifest(CRITICAL_CODE_FILES,ROOT)
    output.mkdir(parents=True,exist_ok=False)
    (output/'code_hashes.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'run_id':run_id,'code_bundle_sha256':bundle['code_bundle_sha256'],
            'original_graphs_untouched':True,'database_accessed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    args=p.parse_args();print(json.dumps(prepare_run(args.run_id),ensure_ascii=False));return 0

if __name__=='__main__': raise SystemExit(main())
