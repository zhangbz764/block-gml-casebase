"""Explicit verification of a new part's ledger and actual graph metadata; no SQL."""
import argparse
import json
import re
from pathlib import Path
from .run_batches import ROOT,PART_IDS,_read_csv_rows,_latest,_metadata
from ..core.export_v03 import read_graph
from ..dataset_io import inside

GRAPH_METADATA_KEYS = (
    "source_manifest_sha256", "full_manifest_sha256", "part_manifest_sha256",
    "encoding_spec_sha256", "encoding_version", "code_bundle_sha256",
    "part_id", "run_id", "parameter_sha256",
)



def completed_graph_problems(record: dict, expected: dict) -> list[str]:
    problems: list[str] = []
    if record.get("encoding_version") != "0.3.2":
        problems.append("top-level encoding_version is not 0.3.2")
    if record.get("validation", {}).get("passed") is not True:
        problems.append("validation.passed is not true")
    metadata = record.get("phase4b", {})
    for key in GRAPH_METADATA_KEYS:
        if metadata.get(key) != expected.get(key):
            problems.append(f"phase4b metadata mismatch: {key}")
    return problems



def verify_part(run_id,part_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id): raise ValueError('unsafe run id')
    if part_id not in PART_IDS: raise ValueError('unknown part id')
    root=ROOT/'results/runs'/run_id
    part_path=ROOT/'results/metadata/encoding/parts'/f'{part_id}_manifest.csv.gz'
    expected=_metadata(part_id,run_id,100,part_path)
    run=root/'parts'/part_id
    if json.loads((run/'preflight.json').read_text(encoding='utf-8')).get('phase4b')!=expected:
        raise ValueError('part preflight differs from current frozen run')
    planned={r['sample_id'] for r in _read_csv_rows(part_path)}
    statuses=_latest(_read_csv_rows(run/'encoding_status.csv'))
    problems=[]
    for sid in sorted(planned-set(statuses)): problems.append({'sample_id':sid,'problem':'interrupted/no ledger entry'})
    for sid in sorted(set(statuses)-planned): problems.append({'sample_id':sid,'problem':'unplanned ledger identity'})
    for sid,row in statuses.items():
        if row['status']=='excluded_no_valid_building':
            if row.get('graph_path'): problems.append({'sample_id':sid,'problem':'excluded entry has graph_path'})
        elif row['status']=='completed':
            try:
                record=read_graph(inside((root/'dataset').resolve(),row['graph_path']))
                for issue in completed_graph_problems(record,expected):
                    problems.append({'sample_id':sid,'problem':issue})
                if record.get('sample',{}).get('sample_id')!=sid: raise ValueError('wrong graph identity')
            except Exception as error: problems.append({'sample_id':sid,'problem':str(error)})
        else: problems.append({'sample_id':sid,'problem':'unsuccessful status: '+row['status']})
    return {'run_id':run_id,'part_id':part_id,'planned':len(planned),'passed':not problems,'problems':problems}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--part-id',required=True,choices=PART_IDS);args=p.parse_args()
    summary=verify_part(args.run_id,args.part_id);print(json.dumps(summary,ensure_ascii=False))
    return 0 if summary['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
