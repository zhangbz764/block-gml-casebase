"""One real case, <=20 evaluations per mode, 300-second parent timeout; NOT formal."""
import argparse
import subprocess
import sys
import time
import unittest
from data_adapter import (ROOT,case_from_graph,case_payload,code_hash,digest,graph_path,index_rows,load_case,
    load_config,make_initial,run_path,sha256,write_json)
from layout_solver import solve_layout
from prepare_experiment import environment


def run(config):
    begin=time.perf_counter();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful(): raise RuntimeError('Unit checks failed; short solve not started')
    rows=index_rows(config);lookup={r['sample_id']:r for r in rows}
    row=lookup[config['priority_cases'][0]] # No search for a more favorable short-check outcome.
    source=graph_path(row['graph_path'],config['graph_root']);case=case_from_graph(load_case(source),config)
    initial=make_initial(case,config['seeds'][0]);directory=run_path('smoke_v01')
    if directory.exists(): raise ValueError('smoke_v01 already exists; preserve prior checks, do not silently rerun')
    results={}
    for mode in ('R','B'):
        budget={'max_evaluations':20,'max_seconds':max(0.,min(60.,280-(time.perf_counter()-begin)))}
        results[mode]=solve_layout(case,initial,mode,budget);write_json(directory/f'{mode}.json',results[mode])
    write_json(directory/'input.json',{'case':case_payload(case),'initial':initial.tolist(),'source':str(source),'sha256':sha256(source)})
    from render_review import comparison_images,original_layout
    from evaluate_generation import verified_layout
    layouts={mode:verified_layout(case,r['layout']) for mode,r in results.items() if r['layout']}
    comparison_images(case,[('Original',original_layout(case)),('Short R',layouts.get('R')),('Short B',layouts.get('B'))],
                      [original_layout(case)]+list(layouts.values()),directory/'short_preview')
    seconds=[r['elapsed_seconds']/max(1,r['evaluations']) for r in results.values()]
    report={'state':'CODE_SHORT_CHECK_ONLY','formal_prepare':False,'formal_solves':0,'real_cases':1,'case_id':case['case_id'],
        'unit_tests':{'run':outcome.testsRun,'failures':len(outcome.failures),'errors':len(outcome.errors)},'code_hash':code_hash(),
        'source_sha256':sha256(source),'config_hash':digest(config),'source_relation_max_errors':case['relation_check'],
        'initial_sha256_both_modes':digest(initial.tolist()),'modes':{mode:{k:r[k] for k in ('evaluations','elapsed_seconds','stop_reason','outcome')} for mode,r in results.items()},
        'seconds_per_evaluation':seconds,'rough_seconds_6000_evaluations':max(seconds)*6000,
        'budget_note':'Short-case timing only; geometry changes can change cost. Formal cap remains 600 seconds per attempt.',
        'elapsed_seconds':time.perf_counter()-begin,'condition_warnings':case['condition_warnings']}
    write_json(ROOT/'results'/'environment.json',environment());write_json(ROOT/'results'/'verification_summary.json',report)
    print(report)
    if any(r['stop_reason']=='exception' or r['layout'] is None for r in results.values()): raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_v01.json');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker: run(load_config(a.config));return
    try: result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'short_check.py'),'--config',str(__import__('pathlib').Path(a.config).resolve()),'--worker'],timeout=300)
    except subprocess.TimeoutExpired:
        write_json(ROOT/'results'/'verification_summary.json',{'state':'SHORT_CHECK_TIMEOUT','formal_solves':0,'time_limit_seconds':300});raise SystemExit(1)
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
