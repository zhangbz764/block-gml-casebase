"""Bounded shared Powell optimizer; ONLY R/B difference is relation weight."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from shapely.affinity import rotate,scale,translate
from data_adapter import RELATIONS,relation_module,to_metric,variable_bounds


def pair_relations(a,b,angle_a,angle_b,site_area):
    implementation=relation_module();difference=implementation._axial_difference_deg(angle_a,angle_b)
    overlap,_=implementation._long_side_relations(a,b,angle_a,angle_b)
    return {'footprint_gap_block_norm':float(a.distance(b)/math.sqrt(site_area)),
            'orientation_similarity':(1+math.cos(2*math.radians(difference)))/2,
            'long_side_overlap_ratio':float(overlap)}


def make_layout(case,variables):
    v=np.asarray(variables,dtype=float).reshape(len(case['geometries']),5)
    if not np.isfinite(v).all() or (v[:,3]<=0).any(): raise ValueError('Invalid layout variables')
    centers=to_metric(v[:,:2],case['frame']);geometries=[]
    for original,target,row in zip(case['geometries'],centers,v,strict=True):
        center=original.centroid;g=scale(original,xfact=row[3],yfact=row[3],origin=(center.x,center.y))
        g=rotate(g,row[2],origin=(center.x,center.y));g=translate(g,xoff=target[0]-center.x,yoff=target[1]-center.y)
        geometries.append(g)
    return {'geometries':geometries,'centers':centers,'angles':(case['angles']+v[:,2])%180,
            'heights':v[:,4].copy(),'variables':v.copy()}


def evaluate_layout(case,layout,mode):
    if mode not in ('R','B'): raise ValueError('Unknown mode')
    cfg=case['config'];geometries=layout['geometries'];heights=np.asarray(layout['heights']);area=sum(g.area for g in geometries)
    outside=sum(g.difference(case['boundary']).area for g in geometries)/area
    overlap=sum(a.intersection(b).area for i,a in enumerate(geometries) for b in geometries[i+1:])/area
    far=sum(g.area*h/cfg['floor_height_m'] for g,h in zip(geometries,heights))/case['boundary'].area
    far_error=abs(far-case['target_far'])/case['target_far'];scales=layout['variables'][:,3]
    common=cfg['feasibility_weight']*(outside+overlap+far_error)+cfg['scale_weight']*float(np.mean((scales-1)**2))
    penalties={f:[] for f in RELATIONS};counts={f:{'valid':0,'skipped':0} for f in RELATIONS};deviations={f:[] for f in RELATIONS}
    for edge in case['edges']:
        i,j=edge['i'],edge['j'];measured=pair_relations(geometries[i],geometries[j],layout['angles'][i],layout['angles'][j],case['boundary'].area)
        for field in RELATIONS:
            reference=edge['reference'][field]
            if reference is None: counts[field]['skipped']+=1;continue
            tolerance=cfg['gap_tolerance_absolute']+cfg['gap_tolerance_relative']*reference if field=='footprint_gap_block_norm' else cfg['relation_tolerance']
            difference=abs(measured[field]-reference);counts[field]['valid']+=1;deviations[field].append(difference)
            penalties[field].append((max(0,difference-tolerance)/tolerance)**2)
    field_losses={f:float(np.mean(p)) if p else None for f,p in penalties.items()}
    relation=float(np.mean([v for v in field_losses.values() if v is not None])) if any(penalties.values()) else 0.
    lo,hi=variable_bounds(case);v=layout['variables'];bounds_valid=bool(((v>=lo-1e-8)&(v<=hi+1e-8)).all())
    height_valid=bool((heights>=cfg['height_min_m']-cfg['height_tolerance_m']).all() and (heights<=case['height_cap']+cfg['height_tolerance_m']).all())
    feasible=bool(bounds_valid and height_valid and outside<=cfg['outside_max'] and overlap<=cfg['overlap_max'] and far_error<=cfg['far_relative_max'])
    original_centers=np.array([[g.centroid.x,g.centroid.y] for g in case['geometries']])
    return {'objective':float(common+(cfg['relation_weight']*relation if mode=='R' else 0)),
            'common_loss':float(common),'relation_loss':relation,'relation_field_losses':field_losses,
            'relation_deviation':{f:float(np.mean(p)) if p else None for f,p in deviations.items()},'relation_counts':counts,
            'equivalent_far':float(far),'target_far':case['target_far'],'far_relative_error':float(far_error),
            'outside_ratio':float(outside),'overlap_ratio':float(overlap),'height_cap_m':case['height_cap'],
            'height_valid':height_valid,'bounds_valid':bounds_valid,'feasible':feasible,
            'center_change_rms_L':float(np.sqrt(np.mean(np.sum((layout['centers']-original_centers)**2,axis=1)))/case['frame']['L'])}


def layout_payload(layout):
    return {'geometry_wkts':[g.wkt for g in layout['geometries']], 'centers':layout['centers'].tolist(),
            'angles':layout['angles'].tolist(),'heights':layout['heights'].tolist(),'variables':layout['variables'].tolist()}


class BudgetStop(Exception): pass


def solve_layout(case,initial,mode,budget):
    started=time.perf_counter();count=0;best_feasible=None;best_violation=None;reason='converged';error=None;optimizer_converged=False
    lo,hi=variable_bounds(case);initial=np.asarray(initial,dtype=float).copy()
    if not ((initial>=lo)&(initial<=hi)).all(): raise ValueError('Paired initial outside variable bounds')
    def objective(flat):
        nonlocal count,best_feasible,best_violation,reason
        if time.perf_counter()-started>=budget['max_seconds']: reason='timeout';raise BudgetStop()
        if count>=budget['max_evaluations']: reason='evaluation_limit';raise BudgetStop()
        count+=1;layout=make_layout(case,flat);diagnostics=evaluate_layout(case,layout,mode)
        candidate={'layout':layout_payload(layout),'diagnostics':diagnostics}
        if diagnostics['feasible']:
            if best_feasible is None or diagnostics['objective']<best_feasible['diagnostics']['objective']: best_feasible=candidate
        elif best_violation is None or (diagnostics['common_loss'],diagnostics['objective']) < (best_violation['diagnostics']['common_loss'],best_violation['diagnostics']['objective']): best_violation=candidate
        return diagnostics['objective']
    try:
        result=minimize(objective,initial.ravel(),method='Powell',bounds=list(zip(lo.ravel(),hi.ravel())),
                        options={'maxfev':budget['max_evaluations'],'xtol':case['config']['xtol'],'ftol':case['config']['ftol']})
        optimizer_converged=bool(result.success)
        reason='converged' if result.success else 'evaluation_limit' if count>=budget['max_evaluations'] else 'solver_stopped'
    except BudgetStop: pass
    except KeyboardInterrupt: reason='interrupted'
    except Exception as exception: reason='exception';error=f'{type(exception).__name__}: {exception}'
    chosen=best_feasible or best_violation
    return {'mode':mode,'evaluations':count,'elapsed_seconds':time.perf_counter()-started,'stop_reason':reason,
        'optimizer_converged':optimizer_converged,'outcome':'feasible_found' if best_feasible else 'no_feasible' if chosen else 'no_candidate',
        'attempt_finished':reason!='interrupted','error':error,'layout':chosen['layout'] if chosen else None,
        'diagnostics':chosen['diagnostics'] if chosen else None,'initial_variables':initial.tolist()}
