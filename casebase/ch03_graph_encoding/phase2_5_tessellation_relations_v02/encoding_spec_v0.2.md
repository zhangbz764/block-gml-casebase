# Graph encoding v0.2 specification

## Scope and provenance

Encoding v0.2 is derived only from the 15 frozen v0.1 graph JSON files listed in `results/input_manifest.csv`. It does not query or write the database and does not modify v0.1 outputs. Every output graph records its source path and SHA-256.

## Morphological cells

- Input geometry: metric block and building WKT from v0.1.
- Building preprocessing: standard validity repair when needed, followed by a `-0.4 m` buffer. Empty results are retained as explicit `empty_after_shrink` failures.
- Boundary sampling: exterior and interior rings are sampled with maximum spacing `0.5 m`.
- Tessellation: point-Voronoi edges are combined with the block boundary and polygonized. Each polygonized piece is assigned to its nearest sampled site, pieces are unioned by `building_id`, and results are clipped to the block.
- No building is moved, jittered, silently deleted, or merged. The edge-line/polygonize route avoids a GEOS failure mode in which dense convex-hull sites can produce an inverted unbounded Voronoi polygon.
- A one-building block is assigned the full block by the explicit `single_building_rule`.
- Adjacency exists only when two cells share a line boundary longer than `1e-6 m`; point contact is not adjacency.

## Stable identifiers and missing values

Cells retain the source `building_id`. Undirected edge endpoints are lexicographically sorted and `edge_id` is `celladj|<source>|<target>`. Missing relation values are JSON `null`, with a Boolean entry in `missing` and a reason in `missing_reasons`; zero is never used as a missing-height substitute.

## Normalization and formulas

- Block-normalized distances divide by `sqrt(block_area_m2)`.
- Axial angular differences are folded into `[0, 90]` degrees.
- `orientation_similarity = (1 + cos(2 * orientation_diff)) / 2`, with the angle converted to radians.
- Long-side measures use each footprint's oriented minimum bounding rectangle and the mean axial direction. Projected overlap is divided by the shorter projected length. Candidate long-side offset is the minimum perpendicular separation among the two long-side positions of each rectangle, divided by `sqrt(mean(footprint_area_i, footprint_area_j))`.
- `log_area_ratio_abs` and `log_height_ratio_abs` are absolute natural logarithms.
- `volume_ratio` is `max(area*height) / min(area*height)` and is only a LoD1 volume proxy.

The complete edge field contract is in `edge_relation_fields.csv`.

## Quality and interpretation boundary

Tessellation adjacency is a candidate spatial-influence graph, not a semantic claim that every edge is architecturally meaningful. Long-distance flags are descriptive review aids only. Winnipeg and Portland remain `out_of_scope_candidate`; Osaka remains a data-quality stress case. No GNN/GAT training or full-database expansion is part of this phase.
