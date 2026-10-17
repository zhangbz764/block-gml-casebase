"""Read-only case inputs; original polygons and fixed node/edge identities."""
from __future__ import annotations
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PureWindowsPath
import re
import sys
from functools import lru_cache
import numpy as np
from shapely import wkt
from shapely.affinity import scale

ROOT=Path(__file__).resolve().parent
RELATIONS=('footprint_gap_block_norm','orientation_similarity','long_side_overlap_ratio')
UPSTREAM=ROOT.parent/'phase2_5_tessellation_relations_v02'/'tessellation_encoding.py'
LEVEL_CASE_IDS=('P4B_IT_BO_000304','P4B_US_BO_000046','P4B_DE_BE_000370')
LEVEL_IDS=('light','medium','strong')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hash():
    return digest({p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))+[ROOT/'experiment_contract_v01.md',UPSTREAM]})


@lru_cache(maxsize=1)
def relation_module():
    # Pure geometry module: importing definitions does not run tessellation or access DB.
    name='_ch03_p6d_relation_geometry';spec=importlib.util.spec_from_file_location(name,UPSTREAM)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def load_config(filename='config_v01.json'):
    filename=Path(filename).resolve();config=read_json(filename)
    for key in ('index','registry','graph_root'): config[key]=str((filename.parent/config[key]).resolve())
    if not 0<config['max_candidate_reads']<=60: raise ValueError('Candidate read budget must be 1..60')
    if config['seeds']!=list(range(20261013,20261018)): raise ValueError('Exactly five frozen paired seeds required')
    if not 0<config['max_evaluations']<=6000 or not 0<config['max_seconds']<=600: raise ValueError('Formal budget exceeds task')
    if not 0<=config['boundary_compression']<1 or config['floor_height_m']<=0: raise ValueError('Invalid conditions')
    if not 0<config['scale_bounds'][0]<=1<=config['scale_bounds'][1]: raise ValueError('Invalid scale bounds')
    if config.get('revision')=='0.2':
        if config.get('level_id') not in LEVEL_IDS or config['selected_case_ids']!=list(LEVEL_CASE_IDS):
            raise ValueError('v0.2 requires the three fixed cases in the agreed order and an explicit level')
    return config


def graph_path(raw,root):
    if not raw or Path(raw).is_absolute() or PureWindowsPath(raw).drive: raise ValueError('Graph path must be relative to Phase4B')
    root=Path(root).resolve();path=(root/raw).resolve();path.relative_to(root);return path


def index_rows(config):
    with gzip.open(config['index'],'rt',encoding='utf-8-sig',newline='') as f:
        return [r for r in csv.DictReader(f) if r['case_status']=='completed' and str(r['audit_passed']).lower()=='true'
                and 8<=float(r['node_count'])<=20]


def load_case(path):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f: graph=json.load(f)
    if graph.get('encoding_version')!='0.3.2' or str(graph.get('validation',{}).get('passed')).lower()!='true':
        raise ValueError('Case is not a validated v0.3.2 Graph')
    return graph


def valid_polygon(text):
    geom=wkt.loads(text)
    if geom.geom_type not in ('Polygon','MultiPolygon') or geom.is_empty or not geom.is_valid or geom.area<=0:
        raise ValueError('Invalid polygon; no repair or rectangular replacement')
    return geom


def case_from_graph(graph,config):
    from layout_solver import pair_relations
    nodes=graph['nodes'];n=len(nodes)
    if not 8<=n<=20: raise ValueError('Case must have 8..20 nodes')
    geometries=[valid_polygon(node['geometry_wkt_metric']) for node in nodes]
    heights=np.array([float(node['height_m']) for node in nodes]);angles=np.array([float(node['orientation_deg']) for node in nodes])
    if not np.isfinite(heights).all() or (heights<=0).any() or not np.isfinite(angles).all(): raise ValueError('Missing height/orientation')
    ids=[str(node['building_id']) for node in nodes]
    if len(set(ids))!=n: raise ValueError('Duplicate node IDs')
    lookup={identifier:i for i,identifier in enumerate(ids)};edges=[];checks={f:[] for f in RELATIONS}
    original=valid_polygon(graph['graph']['block_geometry_wkt_metric']);origin=original.centroid
    for edge in graph['edges']:
        i,j=lookup[str(edge['source'])],lookup[str(edge['target'])]
        if i==j: raise ValueError('Self edge in source')
        reference={}
        measured=pair_relations(geometries[i],geometries[j],angles[i],angles[j],original.area)
        for field in RELATIONS:
            raw=edge.get(field)
            missing=edge.get('missing',{}).get(field) is True or raw is None
            v=None if missing else float(raw)
            if v is not None and not math.isfinite(v): v=None
            reference[field]=v
            if v is not None: checks[field].append(abs(measured[field]-v))
        edges.append({'i':i,'j':j,'reference':reference})
    if any(not values for values in checks.values()): raise ValueError('Each relation needs at least one observed source edge')
    error={field:max(values) for field,values in checks.items()}
    if max(error.values())>config['relation_check_tolerance']: raise ValueError(f'Source relation semantics mismatch: {error}')
    x0,y0,x1,y1=original.bounds;factors=[1.,1.];factors[0 if x1-x0>=y1-y0 else 1]=1-config['boundary_compression']
    boundary=scale(original,xfact=factors[0],yfact=factors[1],origin=(origin.x,origin.y))
    a,b,c,d=boundary.bounds;frame={'center':[(a+c)/2,(b+d)/2],'L':max(c-a,d-b)}
    hcap=float(np.quantile(heights,.9));floor=config['floor_height_m']
    original_far=sum(g.area*h/floor for g,h in zip(geometries,heights))/original.area
    target=original_far*(1+config['far_increase'])
    upper=sum(g.area*config['scale_bounds'][1]**2*hcap/floor for g in geometries)/boundary.area
    warnings=[]
    if hcap<=config['height_min_m']: warnings.append('Height cap must exceed lower bound; choose another case or discuss conditions')
    if target>upper: warnings.append('Target exceeds optimistic capacity; suggest reducing FAR increase to 10%, researcher must decide')
    return {'case_id':str(graph['sample']['sample_id']),'city':graph['sample']['city'],'node_ids':ids,
            'geometries':geometries,'original_boundary':original,'boundary':boundary,'heights':heights,'angles':angles,
            'edges':edges,'config':dict(config),'frame':frame,'boundary_factors':factors,'boundary_origin':[origin.x,origin.y],
            'height_cap':hcap,'original_far':original_far,'target_far':target,'optimistic_far_upper':upper,
            'relation_check':error,'condition_warnings':warnings}


def to_metric(xy,frame): return np.asarray(xy)*frame['L']+frame['center']


def reference_variables(case):
    centers=np.array([[g.centroid.x,g.centroid.y] for g in case['geometries']])
    centers=(centers-case['boundary_origin'])*case['boundary_factors']+case['boundary_origin']
    xy=(centers-case['frame']['center'])/case['frame']['L']
    return np.column_stack([xy,np.zeros(len(xy)),np.ones(len(xy)),case['heights']])


def variable_bounds(case):
    a,b,c,d=case['boundary'].bounds;center=case['frame']['center'];L=case['frame']['L'];cfg=case['config']
    return np.tile([(a-center[0])/L,(b-center[1])/L,-cfg['rotation_bound_deg'],cfg['scale_bounds'][0],cfg['height_min_m']],(len(case['heights']),1)),np.tile([(c-center[0])/L,(d-center[1])/L,cfg['rotation_bound_deg'],cfg['scale_bounds'][1],case['height_cap']],(len(case['heights']),1))


def make_initial(case,seed):
    rng=np.random.default_rng(seed);v=reference_variables(case);n=len(v)
    v[:,:2]+=rng.normal(0,.03,(n,2));v[:,2]+=rng.uniform(-5,5,n);v[:,3]*=rng.uniform(.95,1.05,n);v[:,4]*=rng.uniform(.95,1.05,n)
    lo,hi=variable_bounds(case);return np.clip(v,lo,hi)


def case_payload(case):
    return {**{k:case[k] for k in ('case_id','city','node_ids','edges','config','frame','boundary_factors','boundary_origin',
                 'height_cap','original_far','target_far','optimistic_far_upper','relation_check','condition_warnings')},
        'geometry_wkts':[g.wkt for g in case['geometries']],'heights':case['heights'].tolist(),'angles':case['angles'].tolist(),
        'original_boundary_wkt':case['original_boundary'].wkt,'boundary_wkt':case['boundary'].wkt}


def restore_case(payload):
    case=dict(payload);case['geometries']=[valid_polygon(s) for s in case.pop('geometry_wkts')]
    for key in ('heights','angles'): case[key]=np.asarray(case[key],dtype=float)
    case['original_boundary']=valid_polygon(case.pop('original_boundary_wkt'));case['boundary']=valid_polygon(case.pop('boundary_wkt'))
    return case


def run_path(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',run_id): raise ValueError('Unsafe run-id')
    return ROOT/'results'/'bulk'/run_id


def record_path(run_id):
    run_path(run_id) # share the same traversal-safe identifier validation
    return ROOT/'results'/'runs'/run_id


def require_confirmed(manifest):
    if manifest.get('state')!='CONFIRMED_FROZEN' or not manifest.get('confirmed_at'): raise ValueError('Researcher input confirmation required')


def load_run(run_id,confirmed=True):
    directory=run_path(run_id);manifest=read_json(directory/'manifest.json')
    if manifest['code_hash']!=code_hash(): raise ValueError('Code or relation implementation changed; use a new run-id')
    if confirmed:
        require_confirmed(manifest);freeze=read_json(directory/'freeze.json')
        if sha256(directory/'manifest.json')!=freeze['manifest_sha256']: raise ValueError('Frozen manifest changed')
    for path,expected in manifest['input_sha256'].items():
        if sha256(directory/path)!=expected: raise ValueError(f'Input changed: {path}')
    return directory,manifest
