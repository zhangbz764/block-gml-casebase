"""Recheck saved polygons; preserve all failures, timeouts and unrun attempts."""
import argparse
import csv
import numpy as np
from shapely import wkt
from data_adapter import ROOT,digest,load_run,read_json,record_path,restore_case,sha256,write_json
from layout_solver import evaluate_layout,make_layout


def diversity(variables,config):
    arrays=[np.asarray(v,dtype=float) for v in variables];parent=list(range(len(arrays)));pairs=[]
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(arrays):
        for j in range(i+1,len(arrays)):
            b=arrays[j];center=float(np.sqrt(np.mean(np.sum((a[:,:2]-b[:,:2])**2,axis=1))))
            scale=float(np.max(abs(a[:,3]-b[:,3])));near=center<=config['duplicate_center_rms_max'] and scale<=config['duplicate_scale_max']
            pairs.append({'i':i,'j':j,'center_rms_L':center,'scale_max_difference':scale,'scale_mean_difference':float(np.mean(abs(a[:,3]-b[:,3]))),
                          'height_mae_m':float(np.mean(abs(a[:,4]-b[:,4]))),'near_duplicate':bool(near)})
            if near: parent[find(j)]=find(i)
    return {'different_layouts':len({find(i) for i in parent}),'pairs':pairs,'component_labels':[find(i) for i in range(len(arrays))]}


def planned_rows(planned,records):
    return [{'case_id':case,'mode':mode,'seed':seed,**records.get((case,mode,seed),{'outcome':'not_run','stop_reason':'not_run'})}
            for case,mode,seed in planned]


def verified_layout(case,payload):
    rebuilt=make_layout(case,payload['variables']);polygons=[wkt.loads(s) for s in payload['geometry_wkts']]
    if len(polygons)!=len(rebuilt['geometries']): raise ValueError('Saved footprint count changed')
    for a,b in zip(polygons,rebuilt['geometries']):
        if not a.is_valid or a.symmetric_difference(b).area>max(1e-7,b.area*1e-9): raise ValueError('Saved polygon does not match original-footprint transform')
    if not np.allclose(payload['heights'],rebuilt['heights'],atol=1e-8,rtol=0): raise ValueError('Saved height/variables mismatch')
    # Actual saved geometry is used to independently recompute violations; not saved diagnostics.
    rebuilt['geometries']=polygons;rebuilt['centers']=np.array([[p.centroid.x,p.centroid.y] for p in polygons]);return rebuilt


def paired_summary(rows):
    pairs=[]
    for case_id,seed in sorted({(r['case_id'],r['seed']) for r in rows}):
        lookup={r['mode']:r for r in rows if r['case_id']==case_id and r['seed']==seed}
        r,b=lookup.get('R',{}),lookup.get('B',{})
        paired=all(x.get('outcome','not_run')!='not_run' for x in (r,b))
        pair={'case_id':case_id,'seed':seed,'both_recorded':paired,
              'both_feasible':bool(r.get('feasible') and b.get('feasible')) if paired else None}
        for metric in ('far_relative_error','relation_loss','center_change_rms_L'):
            pair['R_minus_B_'+metric]=r[metric]-b[metric] if metric in r and metric in b else None
        pairs.append(pair)
    return pairs


def evaluate_generation(run_id,render_review=False):
    directory,m=load_run(run_id);cfg=m['config'];selected=m['selected_case_ids'];record_hash=sha256(directory/'manifest.json')
    compact=record_path(run_id);compact.mkdir(parents=True,exist_ok=True)
    lookup={r['case_id']:r for r in m['cases']};records={};layouts={};cases={}
    for case_id in selected:
        cases[case_id]=restore_case(read_json(directory/lookup[case_id]['input_path'])['case'])
        for mode in ('R','B'):
            for seed in cfg['seeds']:
                path=directory/'layouts'/f'{selected.index(case_id)+1:02d}_{mode}_{seed}.json'
                if not path.exists(): continue
                result=read_json(path);key=(case_id,mode,seed)
                source=read_json(directory/lookup[case_id]['input_path'])['initials'][str(seed)]
                if (result['frozen_manifest_sha256']!=record_hash or result['code_hash']!=m['code_hash']
                    or result['initial_sha256']!=digest(source) or digest(result['initial_variables'])!=digest(source)
                    or result['case_id']!=case_id or result['mode']!=mode or result['seed']!=seed): raise ValueError('Saved result identity/paired initial mismatch')
                row={k:result[k] for k in ('outcome','stop_reason','evaluations','elapsed_seconds','optimizer_converged')}
                if result['layout']:
                    try:
                        layout=verified_layout(cases[case_id],result['layout']);d=evaluate_layout(cases[case_id],layout,mode);layouts[key]=layout
                        row.update({k:d[k] for k in ('feasible','equivalent_far','far_relative_error','outside_ratio','overlap_ratio','height_valid','bounds_valid','relation_loss','center_change_rms_L')})
                        if not d['feasible']: row['outcome']='no_feasible'
                        row['relation_deviation']=d['relation_deviation']
                    except (ValueError,KeyError,TypeError) as error: row.update(outcome='invalid_saved_layout',feasible=False,error=str(error))
                records[key]=row
    planned=[(case,mode,seed) for case in selected for mode in ('R','B') for seed in cfg['seeds']]
    rows=planned_rows(planned,records);summaries=[]
    for case_id in selected:
        for mode in ('R','B'):
            feasible=[r for r in rows if r['case_id']==case_id and r['mode']==mode and r.get('feasible')]
            variable_sets=[layouts[(r['case_id'],r['mode'],r['seed'])]['variables'] for r in feasible]
            diagnostic=diversity(variable_sets,cfg)
            write_json(directory/'diversity'/f'{selected.index(case_id)+1:02d}_{mode}.json',{'seeds':[r['seed'] for r in feasible],**diagnostic})
            summaries.append({'case_id':case_id,'mode':mode,'feasible':len(feasible),'nonduplicate_feasible':diagnostic['different_layouts'],
                'position_change_median_L':float(np.median([r['center_change_rms_L'] for r in feasible])) if feasible else None,
                'relation_loss_median':float(np.median([r['relation_loss'] for r in feasible])) if feasible else None,
                'near_duplicate_seed_groups':[[feasible[i]['seed'] for i,label in enumerate(diagnostic['component_labels']) if label==component]
                                             for component in sorted(set(diagnostic['component_labels']))]})
    for row in rows: row.update(run_id=run_id,level_id=m.get('level_id','single_v01'))
    fields=['run_id','level_id','case_id','mode','seed','outcome','stop_reason','evaluations','elapsed_seconds','optimizer_converged','feasible',
        'equivalent_far','far_relative_error','outside_ratio','overlap_ratio','height_valid','bounds_valid','relation_loss','center_change_rms_L','error']
    with (compact/'metrics_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    counts={'planned':len(planned),'recorded':len(records),'not_run':sum(r['outcome']=='not_run' for r in rows),
            'feasible':sum(bool(r.get('feasible')) for r in rows),'failed':sum(r['outcome'] not in ('not_run','feasible_found') for r in rows),
            'timeouts':sum(r['stop_reason']=='timeout' for r in rows),'exceptions':sum(r['stop_reason']=='exception' for r in rows)}
    state='EXECUTED' if len(records)==len(planned) and not any(r['stop_reason']=='interrupted' for r in rows) else 'PARTIAL_NOT_EXECUTED'
    summary={'run_id':run_id,'level_id':m.get('level_id','single_v01'),'state':state,'counts':counts,
             'case_summaries':summaries,'all_rows':rows,'paired_summary':paired_summary(rows),'researcher_judgment':'not_supplied'}
    write_json(directory/'evaluation_summary.json',summary);write_json(compact/'evaluation_summary.json',summary)
    if render_review:
        from render_review import render_results
        render_results(directory,m,rows,layouts,summaries)
    from render_review import condition_title
    (compact/'phase6d_evidence_pack.md').write_text(
        '# Phase6D '+condition_title(m)+'\n\n'+f'状态：{state}；没有自动标为ANALYZED。\n\n计数（timeout与失败等可交叉）：{counts}\n\n'
        +str(summaries)+f'\n\n全部{len(planned)}项含未运行者见同目录metrics_summary.csv；页面results/review/'+run_id+'/visual_review.html。\n\n'
        +'bulk内保存共同输入、全部布局、两两差值和评价摘要（Git忽略）；可按冻结配置再生成，非v0.3.2新Graph。\n\n'
        +'请通过对话判断：条件是否造成重新组织；R是否更能延续预先描述；是否有不同且值得发展的方案。局限：3个目的性案例、固定节点及原轮廓、连续高度、无完整法规检查。\n',encoding='utf-8')
    return {'counts':counts,'state':state}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--render-review',action='store_true')
    a=p.parse_args();print(evaluate_generation(a.run_id,a.render_review))


if __name__=='__main__': main()
