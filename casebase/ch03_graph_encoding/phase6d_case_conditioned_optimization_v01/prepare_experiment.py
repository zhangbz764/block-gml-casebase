"""Manual proposal preview; explicit confirmation freezes three cases and conditions."""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import numpy as np
from data_adapter import (ROOT,RELATIONS,case_from_graph,case_payload,code_hash,digest,graph_path,index_rows,
    load_case,load_config,load_run,make_initial,read_json,record_path,reference_variables,restore_case,run_path,sha256,variable_bounds,write_json)
from layout_solver import evaluate_layout,make_layout


def environment():
    import sys
    return {'python_executable':sys.executable,'versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','shapely','matplotlib')},'gpu_required':False}


def prepare_cases(config,run_id):
    directory=run_path(run_id)
    if directory.exists():
        _,m=load_run(run_id,confirmed=False)
        if m['config_hash']!=digest(config): raise ValueError('Existing run-id has different input/configuration; use another run-id')
        for p,h in m['source_sha256'].items():
            if sha256(p)!=h: raise ValueError('Source index/registry/Graph changed')
        print('Identical prepared inputs retained');return m
    compact=record_path(run_id);compact.mkdir(parents=True,exist_ok=True)
    with Path(config['registry']).open(encoding='utf-8-sig',newline='') as f: registry=list(csv.DictReader(f,delimiter='\t'))
    if not all(any(r['field']==f'edge.{field}' and r['role']=='core_model' for r in registry) for field in RELATIONS): raise ValueError('Relation registry mismatch')
    rows=index_rows(config);lookup={r['sample_id']:r for r in rows}
    requested=config['selected_case_ids']
    if requested and (len(requested)!=3 or len(set(requested))!=3): raise ValueError('Select exactly 3 distinct cases')
    priorities=requested or config['priority_cases']
    candidates=[lookup[s] for s in priorities if s in lookup]
    if requested and len(candidates)!=3: raise ValueError('Requested case absent or outside 8..20 filtered range')
    if not requested:
        seen={r['sample_id'] for r in candidates}
        candidates+=sorted([r for r in rows if r['sample_id'] not in seen],key=lambda r:digest(['6d',r['sample_id']]))
    prepared=[];rejected=[];sources={config[k]:sha256(config[k]) for k in ('index','registry')};files={};reads=0
    for row in candidates[:config['max_candidate_reads']]:
        reads+=1
        try:
            source=graph_path(row['graph_path'],config['graph_root']);graph=load_case(source)
            for key in ('sample_id','city','block_id'):
                if str(graph['sample'][key])!=row[key]: raise ValueError('Graph/index identity mismatch')
            if len(graph['nodes'])!=int(float(row['node_count'])): raise ValueError('Index count mismatch')
            expected=config.get('expected_node_counts',{}).get(row['sample_id'])
            if expected is not None and len(graph['nodes'])!=expected: raise ValueError('Selected case count differs from agreed preview count')
            case=case_from_graph(graph,config)
            inputs={str(seed):make_initial(case,seed).tolist() for seed in config['seeds']}
            initial_diagnostics={seed:evaluate_layout(case,make_layout(case,v),'B') for seed,v in inputs.items()}
            lo,hi=variable_bounds(case);base=np.clip(reference_variables(case),lo,hi)
            mapped=evaluate_layout(case,make_layout(case,base),'B')
            if mapped['feasible']: case['condition_warnings'].append('Mapped unperturbed layout already feasible: discuss substantive adaptation pressure before confirmation')
            case_id=case['case_id'];position=len(prepared);filename=f'inputs/case_{position+1:02d}.json'
            payload={'case':case_payload(case),'initials':inputs,'initial_diagnostics':initial_diagnostics,'mapped_initial_diagnostics':mapped}
            directory.mkdir(parents=True,exist_ok=True);write_json(directory/filename,payload);files[filename]=sha256(directory/filename)
            sources[str(source)]=sha256(source)
            prepared.append({'case_id':case_id,'city':case['city'],'node_count':len(case['node_ids']),'input_path':filename,
                'graph_path':row['graph_path'],'source_sha256':sources[str(source)],'role':'proposed' if position<3 else 'alternative',
                'organization_description':config['organization_descriptions'].get(case_id,''),
                'original_far':case['original_far'],'target_far':case['target_far'],'height_cap_m':case['height_cap'],
                'original_site_area_m2':case['original_boundary'].area,'new_site_area_m2':case['boundary'].area,
                'original_equivalent_floor_area_m2':case['original_far']*case['original_boundary'].area,
                'target_equivalent_floor_area_m2':case['target_far']*case['boundary'].area,
                'target_floor_area_increase':case['target_far']*case['boundary'].area/(case['original_far']*case['original_boundary'].area)-1,
                'optimistic_far_upper':case['optimistic_far_upper'],'condition_warnings':case['condition_warnings'],
                'relation_check':case['relation_check'],'initial_diagnostics':initial_diagnostics,'mapped_initial_diagnostics':mapped})
            if len(prepared)>=(3 if requested else 6): break
        except (ValueError,KeyError,TypeError,OSError) as error:
            rejected.append({'case_id':row['sample_id'],'reason':str(error)})
            if requested: raise ValueError(f'Requested case cannot be prepared: {row["sample_id"]}: {error}') from error
    if len(prepared)<3: raise ValueError(f'Fewer than three usable cases after {reads} reads: {rejected}')
    manifest={'experiment_id':config['experiment_id'],'run_id':run_id,'state':'PREPARED_PENDING_CONFIRMATION',
        'level_id':config.get('level_id','single_v01'),'revision':config.get('revision','0.1'),
        'confirmed_at':None,'config':config,'config_hash':digest(config),'code_hash':code_hash(),'cases':prepared,
        'source_sha256':sources,'input_sha256':files,'candidate_reads':reads,'rejected':rejected,
        'selection_basis':'Researcher-selected fixed cases; no alternatives or resampling' if requested else 'Purposeful proposals from prior 6A/6C manifests, then stable index order; not generated-result selection'}
    write_json(directory/'manifest.json',manifest);write_json(compact/'manifest.json',manifest);write_json(compact/'environment.json',environment())
    from render_review import render_preview
    render_preview(directory,manifest)
    print('Prepared preview only. Researcher must confirm three cases, organization notes and conditions.');return manifest


def confirm(run_id,case_ids=None,notes=None):
    directory,manifest=load_run(run_id,confirmed=False)
    if manifest['state']=='CONFIRMED_FROZEN':
        load_run(run_id);print('Already frozen; no inputs changed');return
    lookup={row['case_id']:row for row in manifest['cases']};selected=case_ids or [r['case_id'] for r in manifest['cases'] if r['role']=='proposed']
    if manifest.get('revision')=='0.2' and selected!=manifest['config']['selected_case_ids']:
        raise ValueError('v0.2 case selection/order is already fixed; cannot replace cases during confirmation')
    if len(selected)!=3 or len(set(selected))!=3 or not set(selected)<=set(lookup): raise ValueError('Confirm exactly three previewed cases')
    notes=notes or {}
    for case_id in selected:
        row=lookup[case_id]
        description=notes.get(case_id,row['organization_description']).strip()
        if not description: raise ValueError(f'Organization description missing: {case_id}; supply --description CASE_ID=text after discussion')
        if row['condition_warnings']: raise ValueError(f'Conditions need discussion: {case_id}: {row["condition_warnings"]}; revise config and prepare new run-id')
        row['organization_description']=description
    if manifest.get('revision')=='0.2':
        descriptions={case_id:lookup[case_id]['organization_description'] for case_id in selected}
        for other_path in (ROOT/'results'/'runs').glob('*/manifest.json'):
            if other_path.parent.name==run_id: continue
            other=read_json(other_path)
            if other.get('revision')=='0.2' and other.get('state')=='CONFIRMED_FROZEN':
                other_notes={r['case_id']:r['organization_description'] for r in other['cases'] if r['case_id'] in selected}
                if other_notes!=descriptions: raise ValueError('Use identical predeclared organization descriptions across the three levels')
    for p,h in manifest['source_sha256'].items():
        if sha256(p)!=h: raise ValueError('Prepared source changed before confirmation')
    manifest.update(state='CONFIRMED_FROZEN',confirmed_at=datetime.now(timezone.utc).isoformat(),selected_case_ids=selected,
                    confirmation_meaning='Researcher explicitly accepted cases, descriptions, conditions, seeds, weights and thresholds')
    write_json(directory/'manifest.json',manifest)
    freeze={'run_id':run_id,'state':'CONFIRMED_FROZEN','manifest_sha256':sha256(directory/'manifest.json'),
            'code_hash':manifest['code_hash'],'config_hash':manifest['config_hash'],'input_sha256':manifest['input_sha256']}
    write_json(directory/'freeze.json',freeze);write_json(record_path(run_id)/'manifest.json',manifest);write_json(record_path(run_id)/'freeze.json',freeze)
    print('Inputs explicitly confirmed and frozen; no formal solve started')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_v01.json');p.add_argument('--run-id',required=True)
    p.add_argument('--confirm-inputs',action='store_true');p.add_argument('--select-cases',nargs=3);p.add_argument('--description',action='append',default=[])
    a=p.parse_args()
    if a.confirm_inputs:
        notes={}
        for text in a.description:
            key,sep,note=text.partition('=')
            if not sep: p.error('--description must be CASE_ID=organization description')
            notes[key]=note
        confirm(a.run_id,a.select_cases,notes)
    else: prepare_cases(load_config(a.config),a.run_id)


if __name__=='__main__': main()
