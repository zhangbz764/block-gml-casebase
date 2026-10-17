"""Offline 2D/3D cards; identical metric scales, camera and height colors per case."""
from html import escape
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from shapely.geometry.polygon import orient
from shapely.ops import triangulate
from data_adapter import ROOT,read_json,restore_case
from layout_solver import make_layout

font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists(): matplotlib.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
matplotlib.rcParams['axes.unicode_minus']=False


def parts(g): return list(g.geoms) if g.geom_type=='MultiPolygon' else [g]


def polygon2d(axis,geometry,color,fill=True):
    for polygon in parts(geometry):
        polygon=orient(polygon,sign=1);vertices=[];codes=[]
        for ring in (polygon.exterior,*polygon.interiors):
            xy=list(ring.coords);vertices.extend(xy);codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(xy)-2)+[PlotPath.CLOSEPOLY])
        axis.add_patch(PathPatch(PlotPath(vertices,codes),facecolor=color if fill else 'none',edgecolor=color if not fill else '#555',lw=.7))


def prism(axis,geometry,height,color):
    for p in parts(geometry):
        faces=[[[x,y,height] for x,y in list(t.exterior.coords)[:-1]] for t in triangulate(p) if p.covers(t)]
        for ring in (p.exterior,*p.interiors):
            coordinates=list(ring.coords)
            for (a,b),(c,d) in zip(coordinates,coordinates[1:]): faces.append([[a,b,0],[c,d,0],[c,d,height],[a,b,height]])
        axis.add_collection3d(Poly3DCollection(faces,facecolor=color,edgecolor='#777',linewidth=.2,alpha=.9))


def comparison_images(case,panels,all_layouts,prefix):
    # all_layouts is shared across ALL starts/modes, including failed candidates.
    geoms=[case['original_boundary'],case['boundary']]+[g for layout in all_layouts for g in layout['geometries']]
    bounds=np.array([g.bounds for g in geoms]);low=bounds[:,:2].min(0);high=bounds[:,2:].max(0)
    middle=(low+high)/2;span=max(high-low)*1.08;top=max(case['height_cap'],*(float(max(l['heights'])) for l in all_layouts),1.)*1.05
    norm=Normalize(0,top);cmap=plt.get_cmap('viridis')
    for dimension in ('2d','3d'):
        for show_edges in (False,True):
            fig=plt.figure(figsize=(5*len(panels),5));axes=[]
            for index,(title,layout) in enumerate(panels):
                axis=fig.add_subplot(1,len(panels),index+1,projection='3d' if dimension=='3d' else None);axes.append(axis)
                if layout:
                    for g,h in zip(layout['geometries'],layout['heights']):
                        if dimension=='2d': polygon2d(axis,g,cmap(norm(h)))
                        else: prism(axis,g,float(h),cmap(norm(h)))
                    if show_edges:
                        for edge in case['edges']:
                            xy=np.asarray([layout['centers'][edge['i']],layout['centers'][edge['j']]])
                            if dimension=='2d': axis.plot(xy[:,0],xy[:,1],color='#999',lw=.6,alpha=.65)
                            else: axis.plot(xy[:,0],xy[:,1],np.zeros(2),color='#999',lw=.6,alpha=.65)
                else: axis.text2D(.1,.5,'No saved candidate',transform=axis.transAxes) if dimension=='3d' else axis.text(.1,.5,'No saved candidate',transform=axis.transAxes)
                for boundary,color in [(case['original_boundary'],'#e28630'),(case['boundary'],'#222')]:
                    if dimension=='2d': polygon2d(axis,boundary,color,False)
                    else:
                        for p in parts(boundary):
                            for ring in (p.exterior,*p.interiors):
                                xy=np.array(ring.coords);axis.plot(xy[:,0],xy[:,1],np.zeros(len(xy)),color=color,lw=.7)
                axis.set_xlim(middle[0]-span/2,middle[0]+span/2);axis.set_ylim(middle[1]-span/2,middle[1]+span/2)
                if dimension=='2d': axis.set_aspect('equal')
                else: axis.set_zlim(0,top);axis.set_box_aspect((span,span,top));axis.view_init(elev=30,azim=-60)
                axis.set_title(title);axis.ticklabel_format(style='plain',useOffset=False)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=3));axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.tick_params(axis='both',labelsize=8)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.65,label='Height (m)')
            fig.savefig(str(prefix)+f'_{dimension}_{int(show_edges)}.png',dpi=110,bbox_inches='tight');plt.close(fig)


def original_layout(case):
    return {'geometries':case['geometries'],'centers':np.array([[g.centroid.x,g.centroid.y] for g in case['geometries']]),'heights':case['heights']}


def picture(relative_prefix):
    return ''.join(f'<img class="view" data-dim="{dim}" data-edges="{edge}" '+('' if dim=='2d' and edge==0 else 'hidden ')+
                   f'src="{escape(str(relative_prefix))}_{dim}_{edge}.png">' for dim in ('2d','3d') for edge in (0,1))


def attempt_caption(row):
    if row['outcome']=='not_run': return 'not_run：尚未运行（不隐去）'
    numbers=' · '.join(f'{k}={row[k]:.5g}' for k in ('equivalent_far','outside_ratio','overlap_ratio') if k in row)
    return f"{row['outcome']} / {row['stop_reason']} · {numbers}"+(' · '+str(row['error']) if row.get('error') else '')


def condition_title(manifest):
    cfg=manifest['config'];level=manifest.get('level_id',cfg.get('level_id','single_v01'))
    actual=[r['target_floor_area_increase'] for r in manifest.get('cases',[]) if 'target_floor_area_increase' in r]
    increase=actual[0] if actual else (1-cfg['boundary_compression'])*(1+cfg['far_increase'])-1
    return (f"{level} · 边界长方向缩窄{cfg['boundary_compression']:.0%} · 等效FAR提高{cfg['far_increase']:.0%} · "
            f"目标总楼面面积提高{increase:.2%}")


def page_html(cards,summary,title='案例关系参照下的条件适应'):
    return ('<!doctype html><meta charset="utf-8"><title>6D case adaptation</title>'
        '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%}section{border-top:1px solid #ccc;margin:20px 0}.identity{display:none}.badge{background:#eee;padding:8px;white-space:pre-wrap}</style>'
        '<h1>'+escape(title)+'</h1><p>橙线：原场地；黑线：新场地。看新要求与预先写明的组织，不以目标下降宣告成功。等效FAR采用连续高度／3m，不是法定容积率；无完整法规检查。灰线是原参照端点，不是新Tessellation邻接。</p>'
        '<button onclick="dim=dim===\'2d\'?\'3d\':\'2d\';update()">二维／三维</button> '
        '<button onclick="edges=1-edges;update()">原参照连线开／关</button> '
        '<button onclick="document.querySelectorAll(\'.identity\').forEach(e=>e.style.display=e.style.display===\'block\'?\'none\':\'block\')">揭示／隐藏R、B身份</button>'
        +''.join(cards)+'<section><h2>全部结果摘要</h2><div class="identity">'+escape(summary)+'</div></section>'
        '<script>let dim="2d",edges=0;function update(){document.querySelectorAll(".view").forEach(e=>e.hidden=e.dataset.dim!==dim||Number(e.dataset.edges)!==edges)}</script>')


def render_preview(directory,manifest):
    output=ROOT/'results'/'review'/manifest['run_id'];output.mkdir(parents=True,exist_ok=True)
    images=directory/'review_images';images.mkdir(exist_ok=True);cards=[]
    for index,row in enumerate(manifest['cases']):
        payload=read_json(directory/row['input_path']);case=restore_case(payload['case']);original=original_layout(case)
        initials=[make_layout(case,v) for v in payload['initials'].values()];all_layouts=[original]+initials
        for seed,layout in zip(payload['initials'],initials):
            prefix=f'preview_{index+1:02d}_{seed}'
            comparison_images(case,[('Original',original),('Paired initial',layout)],all_layouts,images/prefix)
            cards.append('<section><h2>'+escape(row['case_id']+' · '+row['role']+' · '+seed)+'</h2><p>'+
                escape(row['organization_description'] or '组织描述待研究者通过对话提供，不自动分类')+'</p><div class="badge">'+
                escape(str({k:row[k] for k in ('original_far','target_far','target_equivalent_floor_area_m2','target_floor_area_increase','height_cap_m','optimistic_far_upper','condition_warnings') if k in row}))+
                '\n'+escape(str(payload['initial_diagnostics'][seed]))+'</div>'+picture('../../bulk/'+manifest['run_id']+'/review_images/'+prefix)+'</section>')
            # One image card per proposal; all five saved initial diagnostics remain in the manifest.
            cards.append('<p>所有五个共同起点核验：</p><div class="badge">'+escape(str({s:{k:d[k] for k in ('equivalent_far','outside_ratio','overlap_ratio','feasible')}
                for s,d in payload['initial_diagnostics'].items()}))+'</div>')
            break
    (output/'input_preview.html').write_text(page_html(cards,'Pending researcher confirmation; no formal solve',condition_title(manifest)),encoding='utf-8')


def render_results(directory,manifest,rows,layouts,summaries):
    output=ROOT/'results'/'review'/manifest['run_id'];output.mkdir(parents=True,exist_ok=True)
    images=directory/'review_images';images.mkdir(exist_ok=True);cards=[];cfg=manifest['config']
    for index,case_id in enumerate(manifest['selected_case_ids']):
        entry=next(r for r in manifest['cases'] if r['case_id']==case_id);payload=read_json(directory/entry['input_path']);case=restore_case(payload['case'])
        original=original_layout(case);initials={seed:make_layout(case,payload['initials'][str(seed)]) for seed in cfg['seeds']}
        all_layouts=[original]+list(initials.values())+[v for k,v in layouts.items() if k[0]==case_id]
        prefix=f'original_{index+1:02d}';comparison_images(case,[('Original',original)],all_layouts,images/prefix)
        cards.append('<section><h2>'+escape(case_id)+'</h2><p>'+escape(entry['organization_description'])+'</p>'+picture('../../bulk/'+manifest['run_id']+'/review_images/'+prefix)+'</section>')
        order=('R','B') if index%2==0 else ('B','R')
        for seed in cfg['seeds']:
            prefix=f'pair_{index+1:02d}_{seed}';panels=[('Paired initial',initials[seed])]+[(f'Option {i+1}',layouts.get((case_id,mode,seed))) for i,mode in enumerate(order)]
            comparison_images(case,panels,all_layouts,images/prefix)
            captions=[attempt_caption(next(r for r in rows if (r['case_id'],r['mode'],r['seed'])==(case_id,mode,seed))) for mode in order]
            duplicate_notes={s['mode']:s['near_duplicate_seed_groups'] for s in summaries if s['case_id']==case_id}
            cards.append('<section><h2>'+str(seed)+'</h2><div class="badge">'+escape('\n'.join(f'Option {i+1}: {text}' for i,text in enumerate(captions)))+
                f'\nTarget FAR={case["target_far"]:.5g}, Hcap={case["height_cap"]:.5g}m</div><div class="identity">'+
                escape(str({'option_order':order,'near_duplicate_feasible_seed_groups':duplicate_notes}))+'</div>'+picture('../../bulk/'+manifest['run_id']+'/review_images/'+prefix)+'</section>')
    (output/'visual_review.html').write_text(page_html(cards,str(summaries),condition_title(manifest)),encoding='utf-8')
