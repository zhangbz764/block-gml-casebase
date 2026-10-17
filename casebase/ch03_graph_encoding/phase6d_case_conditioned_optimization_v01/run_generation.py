"""Manual 3x2x5 experiment; requires explicit frozen input confirmation."""
import argparse
import numpy as np
from data_adapter import ROOT,digest,load_run,read_json,restore_case,sha256,write_json
from layout_solver import solve_layout


def run_generation(run_id,case_selector='all',mode='both',resume=False):
    directory,manifest=load_run(run_id);selected=manifest['selected_case_ids'];cfg=manifest['config']
    if case_selector!='all' and case_selector not in selected: raise ValueError('Case not among three frozen selections')
    cases=[r for r in manifest['cases'] if r['case_id'] in selected and (case_selector=='all' or r['case_id']==case_selector)]
    modes=('R','B') if mode=='both' else (mode,);bad=False
    frozen_hash=sha256(directory/'manifest.json')
    # Whole invocation preflight: never run a few attempts then discover a conflicting old result.
    for row in cases:
        for role in modes:
            for seed in cfg['seeds']:
                path=directory/'layouts'/f'{selected.index(row["case_id"])+1:02d}_{role}_{seed}.json'
                if path.exists():
                    old=read_json(path)
                    if old['frozen_manifest_sha256']!=frozen_hash: raise ValueError('Existing result input mismatch')
                    if not resume: raise ValueError('Existing attempt; use explicit --resume to skip completed identical settings')
                    if old['attempt_finished'] and (old['outcome']!='feasible_found' or old['stop_reason'] in ('exception','interrupted')): bad=True
    for row in cases:
        payload=read_json(directory/row['input_path']);case=restore_case(payload['case'])
        for seed in cfg['seeds']:
            initial=np.asarray(payload['initials'][str(seed)],dtype=float)
            for role in modes:
                path=directory/'layouts'/f'{selected.index(row["case_id"])+1:02d}_{role}_{seed}.json'
                if path.exists() and read_json(path)['attempt_finished']: continue
                result=solve_layout(case,initial,role,{'max_evaluations':cfg['max_evaluations'],'max_seconds':cfg['max_seconds']})
                result.update(case_id=row['case_id'],seed=seed,initial_sha256=digest(initial.tolist()),
                    frozen_manifest_sha256=frozen_hash,code_hash=manifest['code_hash'])
                write_json(path,result)
                print(f'{row["case_id"]} {role} {seed}: {result["outcome"]}, {result["stop_reason"]}',flush=True)
                if result['outcome']!='feasible_found' or result['stop_reason'] in ('exception','interrupted'): bad=True
                if result['stop_reason']=='interrupted': return 130
    # One compact table; renderer/evaluation includes every planned or missing attempt.
    from evaluate_generation import evaluate_generation
    evaluate_generation(run_id,False)
    return 1 if bad else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--case',default='all')
    p.add_argument('--mode',choices=('both','R','B'),default='both');p.add_argument('--resume',action='store_true')
    a=p.parse_args();raise SystemExit(run_generation(a.run_id,a.case,a.mode,a.resume))


if __name__=='__main__': main()
