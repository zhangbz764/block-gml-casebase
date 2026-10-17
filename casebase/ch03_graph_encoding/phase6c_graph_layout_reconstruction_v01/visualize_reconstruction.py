"""Pre-render four comparable 2D/3D panels. Never repair predictions."""
from html import escape
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from shapely import wkt
from shapely.geometry.polygon import orient
from data_adapter import ROOT, block_geometry, read_graph, write_json
from evaluate_reconstruction import rectangle_polygons, restore_layout
from train_reconstruction import read_cache


def parts(geom): return list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]


def draw_polygon(axis, geom, color, fill=True):
    for polygon in parts(geom):
        polygon = orient(polygon, sign=1.)
        vertices, codes = [], []
        for ring in (polygon.exterior, *polygon.interiors):
            xy = list(ring.coords); vertices.extend(xy)
            codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(xy)-2)+[MplPath.CLOSEPOLY])
        axis.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color if fill else 'none', edgecolor='#555', lw=.5, alpha=.85))


def prism(axis, geom, height, color):
    for polygon in parts(geom):
        # Triangulated top retains original concavities/holes; no geometric repair.
        from shapely.ops import triangulate
        faces = []
        for triangle in triangulate(polygon):
            if polygon.covers(triangle): faces.append([[x, y, height] for x, y in list(triangle.exterior.coords)[:-1]])
        for ring in (polygon.exterior, *polygon.interiors):
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                faces.append([[x0, y0, 0], [x1, y1, 0], [x1, y1, height], [x0, y0, height]])
        axis.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor='#777', linewidth=.15, alpha=.85))


def render_case(record, target, relation, baseline, frame, height_stats, output_prefix):
    boundary = block_geometry(record)
    original = [wkt.loads(n['geometry_wkt_metric']) for n in record['nodes']]
    heights = [float(n['height_m']) for n in record['nodes']]
    layouts = [restore_layout(t, frame, height_stats) for t in (target, relation, baseline)]
    panels = [(original, heights)]+[(rectangle_polygons(layout), layout[:, 6]) for layout in layouts]
    titles = ['Original footprint / LoD1', 'Rectangle target', 'A: relation', 'B: building set']
    all_geometries = [boundary]+[g for geometries, _ in panels for g in geometries]
    bounds = np.asarray([g.bounds for g in all_geometries]); lo = bounds[:, :2].min(0); hi = bounds[:, 2:].max(0)
    middle = (lo+hi)/2; span = max(hi-lo)*1.08; xlim = (middle[0]-span/2, middle[0]+span/2); ylim = (middle[1]-span/2, middle[1]+span/2)
    all_heights = np.concatenate([np.asarray(h) for _, h in panels]); bottom = min(0., float(all_heights.min())); top = max(1., float(all_heights.max()))
    norm = Normalize(vmin=bottom, vmax=top); cmap = plt.get_cmap('viridis')
    for mode in ('2d', '3d'):
        fig = plt.figure(figsize=(16, 4.7)); axes = []
        for k, ((geometries, hs), title) in enumerate(zip(panels, titles, strict=True)):
            axis = fig.add_subplot(1, 4, k+1, projection='3d' if mode=='3d' else None); axes.append(axis)
            for geom, height in zip(geometries, hs, strict=True):
                if mode=='2d': draw_polygon(axis, geom, cmap(norm(height)))
                else: prism(axis, geom, float(height), cmap(norm(height)))
            if mode=='2d':
                draw_polygon(axis, boundary, '#222', fill=False); axis.set_aspect('equal')
            else:
                for polygon in parts(boundary):
                    for ring in (polygon.exterior, *polygon.interiors):
                        xy = np.asarray(ring.coords); axis.plot(xy[:, 0], xy[:, 1], np.zeros(len(xy)), color='#222', lw=.6)
                axis.set_zlim(bottom, top*1.05); axis.view_init(elev=30, azim=-60)
                axis.set_box_aspect((span, span, top-bottom))
            axis.set_xlim(xlim); axis.set_ylim(ylim); axis.set_title(title); axis.ticklabel_format(useOffset=False, style='plain')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=.7, label='height (m)')
        fig.savefig(str(output_prefix)+f'_{mode}.png', dpi=130, bbox_inches='tight'); plt.close(fig)


def render_review_page(run_id, selected, entries, predictions, directory, config, height_stats):
    output = ROOT/'results'/'review'/run_id; images = output/'images'; images.mkdir(parents=True, exist_ok=True)
    lookup = {r['sample_id']: r for r in entries}; cards = []
    for row in selected:
        identifier = row['sample_id']; entry = lookup[identifier]
        if entry['split'] != 'test': raise ValueError('Only test cases may enter review')
        record = read_graph(entry, config['graph_root']); sample = read_cache(directory, entry)
        # Use numeric filenames, never IDs as filesystem paths.
        name = f'case_{len(cards)+1:02d}'
        render_case(record, sample['target'], predictions[identifier]['relation'], predictions[identifier]['set'],
                    entry['frame'], height_stats, images/name)
        descriptions = []
        for kind in ('relation', 'set'):
            descriptions.append(f"{kind}: center {row[kind+'_center_error_m']:.2f}m; long/short "
                f"{row[kind+'_long_relative_error']:.3f}/{row[kind+'_short_relative_error']:.3f}; height {row[kind+'_height_mae_m']:.2f}m")
        cards.append(f'<section><h2>{escape(identifier)} · N={int(float(row["node_count"]))}</h2>'
            f'<p>{escape(row["selection_source"])} · {escape(" | ".join(descriptions))} · A−B center/L={row["A_minus_B_center_error_norm"]:.4f}</p>'
            f'<img class="view2d" src="images/{name}_2d.png"><img class="view3d" hidden src="images/{name}_3d.png"></section>')
    html = '<!doctype html><meta charset="utf-8"><title>6C test reconstruction</title><style>body{font-family:sans-serif;margin:24px}img{width:100%}section{border-top:1px solid #ccc}</style>'
    html += '<h1>布局与体量重构：测试案例</h1><p>看布局和体量是否恢复，不是看人工变化是否发生。边界和建筑数量已知；未作几何修复。已评价不等于研究者接受。</p>'
    html += '<button onclick="setMode(false)">二维</button> <button onclick="setMode(true)">三维</button>'+''.join(cards)
    html += '<script>function setMode(d){document.querySelectorAll(".view2d").forEach(e=>e.hidden=d);document.querySelectorAll(".view3d").forEach(e=>e.hidden=!d)}</script>'
    (output/'visual_review.html').write_text(html, encoding='utf-8')
    write_json(output/'selection_manifest.json', {'run_id': run_id, 'cases': selected})
