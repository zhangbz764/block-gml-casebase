"""Compact per-case/per-level comparison; missing runs are not measured failures."""
import argparse
import csv
from html import escape
import json
from data_adapter import ROOT,LEVEL_CASE_IDS,LEVEL_IDS,read_json,record_path,run_path
from evaluate_generation import paired_summary


def summarize_levels(run_ids):
    levels={};rows=[]
    for run_id in run_ids:
        path=record_path(run_id)/'manifest.json'
        if path.exists():
            manifest=read_json(path);cfg=manifest['config'];level=manifest.get('level_id',cfg.get('level_id'))
        else:
            if not run_id.startswith('adapt_v02_') or run_id.removeprefix('adapt_v02_') not in LEVEL_IDS:
                raise ValueError('Missing nonstandard run: cannot invent its selected cases or conditions')
            level=run_id.removeprefix('adapt_v02_');cfg=read_json(ROOT/f'config_{level}_v02.json');manifest=None
        if level not in LEVEL_IDS or level in levels: raise ValueError('Supply at most one run per light/medium/strong level')
        selected=(manifest or {}).get('selected_case_ids') or cfg['selected_case_ids'];seeds=cfg['seeds']
        if selected!=list(LEVEL_CASE_IDS): raise ValueError('Cross-level fixed case identities/order mismatch')
        evaluation_path=record_path(run_id)/'evaluation_summary.json'
        evaluation=read_json(evaluation_path) if evaluation_path.exists() else {}
        if evaluation and (evaluation.get('run_id')!=run_id or evaluation.get('level_id')!=level):
            raise ValueError('Evaluation belongs to another run/level')
        planned={(case,mode,seed) for case in selected for mode in ('R','B') for seed in seeds}
        entries=evaluation.get('all_rows',[]);seen=set()
        for entry in entries:
            key=(entry['case_id'],entry['mode'],entry['seed'])
            if key not in planned or key in seen: raise ValueError('Unexpected or duplicate planned attempt')
            seen.add(key)
        summaries={(s['case_id'],s['mode']):s for s in evaluation.get('case_summaries',[])}
        all_pairs=paired_summary(entries)
        details={r['case_id']:r for r in (manifest or {}).get('cases',[])}
        for case in selected:
            for mode in ('R','B'):
                group=[r for r in entries if r['case_id']==case and r['mode']==mode and r['outcome']!='not_run']
                observed=len(group);finished=sum(r['stop_reason']!='interrupted' for r in group)
                summary=summaries.get((case,mode),{})
                detail=details.get(case,{})
                row={'case_id':case,'city':detail.get('city',''),'level_id':level,'run_id':run_id,
                    'boundary_compression':cfg['boundary_compression'],'far_increase':cfg['far_increase'],
                    'target_floor_area_increase':detail.get('target_floor_area_increase',(1-cfg['boundary_compression'])*(1+cfg['far_increase'])-1),
                    'mode':mode,'planned':len(seeds),'recorded_count':observed,'not_run_count':len(seeds)-observed,
                    'status':'未运行' if not observed else '已执行' if finished==len(seeds) else '部分完成',
                    'feasible_count':sum(bool(r.get('feasible')) for r in group) if observed else None,
                    'nonduplicate_feasible':summary.get('nonduplicate_feasible') if observed else None,
                    'failed_count':sum(r['outcome']!='feasible_found' for r in group) if observed else None,
                    'timeout_count':sum(r['stop_reason']=='timeout' for r in group) if observed else None,
                    'exception_count':sum(r['stop_reason']=='exception' for r in group) if observed else None,
                    'paired_summary':[pair for pair in all_pairs if pair['case_id']==case]}
                rows.append(row)
        levels[level]={'run_id':run_id,'input_state':(manifest or {}).get('state','NOT_PREPARED'),
            'cases':selected,'seeds':seeds,'planned':len(planned),'observed':sum(r['outcome']!='not_run' for r in entries)}
    rows.sort(key=lambda r:(list(LEVEL_CASE_IDS).index(r['case_id']),list(LEVEL_IDS).index(r['level_id']),('R','B').index(r['mode'])))
    counts={'planned':sum(r['planned'] for r in rows),'recorded':sum(r['recorded_count'] for r in rows),
            'not_run':sum(r['not_run_count'] for r in rows),
            'failed_observed':sum(r['failed_count'] or 0 for r in rows),'timeouts_observed':sum(r['timeout_count'] or 0 for r in rows)}
    return {'rows':rows,'levels':levels,'counts':counts,'count_note':'Timeouts/exceptions may overlap failures; unrun is not a measured zero or failure'}


def write_outputs(summary,render_review=False):
    destination=ROOT/'results';destination.mkdir(exist_ok=True)
    with (destination/'levels_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary['rows'][0]));writer.writeheader()
        for row in summary['rows']: writer.writerow({**row,'paired_summary':json.dumps(row['paired_summary'],ensure_ascii=False)})
    if not render_review: return
    output=destination/'review'/'levels_v02';output.mkdir(parents=True,exist_ok=True);links=[]
    for level in LEVEL_IDS:
        if level not in summary['levels']: continue
        info=summary['levels'][level];run_id=info['run_id'];items=[]
        for filename,label in [('input_preview.html','输入预览'),('visual_review.html','全部结果')]:
            path=destination/'review'/run_id/filename
            items.append(f'<a href="../{escape(run_id)}/{filename}">{label}</a>' if path.exists() else label+'未生成')
        links.append('<p>'+escape(level+' · '+run_id+' · '+info['input_state'])+'：'+' / '.join(items)+'</p>')
    headers=['案例','档位','边界缩窄','等效FAR提高','目标总楼面面积提高','版本','状态','记录／计划','可行','不同可行','失败','超时']
    table=[]
    for row in summary['rows']:
        values=[row['case_id'],row['level_id'],f"{row['boundary_compression']:.0%}",f"{row['far_increase']:.0%}",
            f"{row['target_floor_area_increase']:.2%}",row['mode'],row['status'],f"{row['recorded_count']}/{row['planned']}",
            row['feasible_count'],row['nonduplicate_feasible'],row['failed_count'],row['timeout_count']]
        table.append('<tr>'+''.join('<td>'+escape('未测' if value is None else str(value))+'</td>' for value in values)+'</tr>')
    html='<!doctype html><meta charset="utf-8"><title>6D 三档总览</title><style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>'
    html+='<h1>6D 轻／中／强三档总览</h1><p>三档同时改变边界与等效FAR，只能讨论组合压力。未运行不是实测0；失败和超时可以交叉计数。复用单档页面，不新增图集。</p>'
    html+='<p>'+escape(str(summary['counts']))+'</p>'+''.join(links)+'<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join(table)+'</table>'
    (output/'levels_overview.html').write_text(html,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-ids',nargs='+',required=True);p.add_argument('--render-review',action='store_true')
    args=p.parse_args();summary=summarize_levels(args.run_ids);write_outputs(summary,args.render_review);print(summary['counts'])


if __name__=='__main__': main()
