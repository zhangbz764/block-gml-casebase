"""Render only the bounded Phase 3.5A review manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from psycopg2 import sql
from shapely.geometry import shape


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
sys.path.insert(0, str(CH03_DIR))

from shared.db_readonly import readonly_connection
from run_phase3_5a import validate_run_id


def build_review_html(blocks: dict[str, Any], buildings: dict[str, Any]) -> str:
    """Build a self-contained data page that uses Leaflet only for display."""
    block_json = json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))
    building_json = json.dumps(buildings, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 3.5A review cases</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{{height:100%;margin:0}} .leaflet-popup-content{{max-height:260px;overflow:auto}} table{{border-collapse:collapse}}td{{padding:2px 5px;border-bottom:1px solid #ddd}}</style>
</head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const blocks={block_json}; const buildings={building_json};
const map=L.map('map'); L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(map);
function esc(value){{return String(value??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));}}
function popup(feature){{return '<table>'+Object.entries(feature.properties||{{}}).map(([k,v])=>'<tr><td>'+esc(k)+'</td><td>'+esc(v)+'</td></tr>').join('')+'</table>';}}
const buildingLayer=L.geoJSON(buildings,{{style:{{color:'#777',weight:0.6,fillColor:'#bdbdbd',fillOpacity:0.55}}}}).addTo(map);
const blockLayer=L.geoJSON(blocks,{{style:{{color:'#d7301f',weight:2,fillOpacity:0.05}},onEachFeature:(f,l)=>l.bindPopup(popup(f))}}).addTo(map);
const bounds=blockLayer.getBounds(); if(bounds.isValid()) map.fitBounds(bounds.pad(0.05)); else map.setView([0,0],2);
</script></body></html>"""


def _fetch_geojson(connection, manifest: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = [(str(row.city), str(row.block_id)) for row in manifest.itertuples(index=False)]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT city, block_id, ST_AsGeoJSON(geom)
            FROM summary.lod1_candidate_blocks
            WHERE (city, block_id) IN (SELECT * FROM unnest(%s::text[], %s::text[]))
            ORDER BY city, block_id
            """,
            ([item[0] for item in keys], [item[1] for item in keys]),
        )
        block_rows = cursor.fetchall()
        cursor.execute(
            "SELECT city, city_query FROM summary.city_name_map WHERE city = ANY(%s) ORDER BY city",
            (sorted({item[0] for item in keys}),),
        )
        city_queries = dict(cursor.fetchall())

    properties = {
        (str(row.city), str(row.block_id)): {
            key: value
            for key, value in row._asdict().items()
            if key in {
                "city", "block_id", "machine_status", "hard_reasons", "scope_reasons",
                "review_sample_roles", "area_m2", "elongation", "compactness",
                "lod1_building_count", "bcr", "extended_case_no", "sample_origin",
                "count_band", "area_band",
            }
        }
        for row in manifest.itertuples(index=False)
    }
    block_features = [
        {
            "type": "Feature",
            "geometry": json.loads(geometry),
            "properties": properties[(str(city), str(block_id))],
        }
        for city, block_id, geometry in block_rows
    ]
    building_features = []
    for city, group in manifest.groupby("city", sort=True):
        city_query = city_queries[str(city)]
        relation = f"{city_query}_buildings_lod1"
        with connection.cursor() as cursor:
            statement = sql.SQL(
                "SELECT block_id, building_id, ST_AsGeoJSON(geom_2d) FROM lod1.{} WHERE block_id = ANY(%s) ORDER BY block_id, building_id"
            ).format(sql.Identifier(relation))
            cursor.execute(statement, (group["block_id"].astype(str).tolist(),))
            for block_id, building_id, geometry in cursor.fetchall():
                building_features.append(
                    {
                        "type": "Feature",
                        "geometry": json.loads(geometry),
                        "properties": {"city": str(city), "block_id": str(block_id), "building_id": str(building_id)},
                    }
                )
    return (
        {"type": "FeatureCollection", "features": block_features},
        {"type": "FeatureCollection", "features": building_features},
    )


def _plot_geometry(ax, geometry, *, color: str, linewidth: float, fill: bool = False) -> None:
    if geometry.geom_type == "Polygon":
        x, y = geometry.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)
        if fill:
            ax.fill(x, y, color=color, alpha=0.35)
    elif geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            _plot_geometry(ax, part, color=color, linewidth=linewidth, fill=fill)


def _case_title(case: Any, *, paginated: bool) -> str:
    reasons = case.hard_reasons if case.hard_reasons else case.scope_reasons
    if not paginated:
        return f"{case.city} | {case.block_id}\n{reasons}"
    building_count = f"{float(case.lod1_building_count):g}"
    area_m2 = f"{float(case.area_m2):g}"
    return (
        f"#{case.extended_case_no} {case.city} | {case.block_id}\n"
        f"buildings={building_count} | area={area_m2} m² | {case.sample_origin}"
    )


def _render_contact_sheets(
    manifest: pd.DataFrame,
    blocks: dict[str, Any],
    buildings: dict[str, Any],
    output_dir: Path,
    *,
    page_size: int | None = None,
) -> None:
    if page_size is not None and page_size < 1:
        raise ValueError("page_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    block_by_key = {
        (str(feature["properties"]["city"]), str(feature["properties"]["block_id"])): feature
        for feature in blocks["features"]
    }
    buildings_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for feature in buildings["features"]:
        key = (str(feature["properties"]["city"]), str(feature["properties"]["block_id"]))
        buildings_by_key.setdefault(key, []).append(feature)
    for rule in (*[f"H{i:02d}" for i in range(3, 6)], *[f"S{i:02d}" for i in range(1, 7)]):
        rule_cases = manifest.loc[manifest[rule].astype("string").str.lower().eq("true")]
        if rule_cases.empty:
            continue
        if page_size is None:
            pages = [rule_cases.head(12)]
        else:
            pages = [rule_cases.iloc[start:start + page_size] for start in range(0, len(rule_cases), page_size)]
        for page_index, cases in enumerate(pages, start=1):
            columns = 4
            rows = math.ceil(len(cases) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, case in zip(axes.ravel(), cases.itertuples(index=False)):
                key = (str(case.city), str(case.block_id))
                block = block_by_key.get(key)
                if block:
                    _plot_geometry(ax, shape(block["geometry"]), color="#d7301f", linewidth=1.5)
                for feature in buildings_by_key.get(key, []):
                    _plot_geometry(ax, shape(feature["geometry"]), color="#777777", linewidth=0.35, fill=True)
                ax.set_title(_case_title(case, paginated=page_size is not None), fontsize=8)
                ax.set_aspect("equal", adjustable="datalim")
                ax.axis("off")
            fig.tight_layout()
            filename = (
                f"{rule}_contact_sheet.png"
                if page_size is None
                else f"{rule}_contact_sheet_p{page_index:02d}.png"
            )
            fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--page-size", type=int)
    args = parser.parse_args()
    validate_run_id(args.run_id)
    manifest_path = RESULTS_DIR / "audit" / args.run_id / "review_manifest.csv"
    output_dir = RESULTS_DIR / "visual_review" / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite visual review output: {output_dir}")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", keep_default_na=False)
    if len(manifest) > 72:
        raise RuntimeError(f"Review manifest exceeds 72 cases: {len(manifest)}")
    with readonly_connection(statement_timeout_ms=600_000) as connection:
        blocks, buildings = _fetch_geojson(connection, manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "review_cases.html").write_text(
        build_review_html(blocks, buildings), encoding="utf-8"
    )
    _render_contact_sheets(
        manifest,
        blocks,
        buildings,
        output_dir / "contact_sheets",
        page_size=args.page_size,
    )
    print(json.dumps({"run_id": args.run_id, "case_count": len(manifest), "output": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
