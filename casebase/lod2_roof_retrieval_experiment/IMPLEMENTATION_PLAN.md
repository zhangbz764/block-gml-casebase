# Luxembourg LOD2 Roof Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook-led, reproducible Luxembourg experiment that derives G1/M1 and G2/M2 metrics from the same LOD2 buildings, writes only to an isolated experiment schema, and produces deterministic development-query Top-5 retrieval results.

**Architecture:** Pure geometry, aggregation, and retrieval logic live in small Python modules with `unittest` coverage. A database adapter reads existing `block`, `lod2`, and `summary` objects but permits writes only to `exp_lod2_roof_lu`. A single output-free notebook orchestrates schema creation, metric computation, quality summaries, file exports, query selection, and retrieval.

**Tech Stack:** Python 3 in Conda environment `urbanCode`; NumPy, Pandas, SciPy, scikit-learn, Shapely, psycopg2, nbformat, PyArrow; PostgreSQL/PostGIS; Jupyter Notebook.

**Spec:** `casebase/lod2_roof_retrieval_experiment/EXPERIMENT_DESIGN.md`

## Global Constraints

- Source city is Luxembourg only.
- Existing `block`, `lod1`, `lod2`, and `summary` schemas are read-only.
- All database DDL/DML targets `exp_lod2_roof_lu` and exactly these experiment tables: `building_metrics`, `block_metrics`, and `retrieval_results`.
- All filesystem outputs stay under `casebase/lod2_roof_retrieval_experiment/outputs/`.
- The committed notebook contains no cell outputs and uses helper modules rather than duplicating method implementations.
- No new Python dependencies are installed; tests use the standard-library `unittest` runner.

---

### Task 1: 3D roof geometry kernel

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/roof_geometry.py`
- Create: `casebase/lod2_roof_retrieval_experiment/tests/__init__.py`
- Create: `casebase/lod2_roof_retrieval_experiment/tests/test_roof_geometry.py`

**Interfaces:**
- Consumes: Shapely `Polygon` objects whose XY coordinates are already in Luxembourg's metric CRS and whose Z coordinates are metres.
- Produces: `FaceMetrics`, `triangulate_polygon_3d`, `measure_polygon_3d`, `ground_elevation`, `g1_height`, and `roof_relief_ratio`.

- [ ] **Step 1: Write failing synthetic-geometry tests**

```python
import math
import unittest
from shapely.geometry import Polygon

from roof_geometry import g1_height, measure_polygon_3d, roof_relief_ratio


class RoofGeometryTests(unittest.TestCase):
    def test_flat_roof(self):
        roof = Polygon([(0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)])
        result = measure_polygon_3d(roof)
        self.assertAlmostEqual(result.projected_area, 100.0, places=6)
        self.assertAlmostEqual(result.area_3d, 100.0, places=6)
        self.assertAlmostEqual(result.mean_slope_deg, 0.0, places=6)
        self.assertAlmostEqual(result.mean_z, 10.0, places=6)

    def test_forty_five_degree_roof(self):
        roof = Polygon([(0, 0, 0), (10, 0, 10), (10, 10, 10), (0, 10, 0)])
        result = measure_polygon_3d(roof)
        self.assertAlmostEqual(result.projected_area, 100.0, places=6)
        self.assertAlmostEqual(result.area_3d, 100.0 * math.sqrt(2), places=6)
        self.assertAlmostEqual(result.mean_slope_deg, 45.0, places=6)
        self.assertAlmostEqual(result.mean_z, 5.0, places=6)

    def test_g1_height_and_relief(self):
        roof = Polygon([(0, 0, 8), (10, 0, 13), (10, 10, 13), (0, 10, 8)])
        ground = Polygon([(0, 0, 1), (10, 0, 1), (10, 10, 1), (0, 10, 1)])
        self.assertAlmostEqual(g1_height([roof], [ground]), 9.5, places=6)
        self.assertAlmostEqual(roof_relief_ratio([roof], [ground]), 5.0 / 12.0, places=6)
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_roof_geometry -v`

Expected: FAIL because `roof_geometry` does not exist.

- [ ] **Step 3: Implement metric-CRS triangulation and measurements**

Implement `FaceMetrics` as a frozen dataclass with `projected_area`, `area_3d`, `mean_slope_deg`, and `mean_z`. Triangulate the polygon in XY with `shapely.ops.triangulate`, retain triangles whose representative point is covered by the polygon, map triangle XY vertices to original Z values, and aggregate:

```python
area_3d = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
area_xy = abs(np.cross((p1 - p0)[:2], (p2 - p0)[:2])) / 2.0
slope = np.degrees(np.arccos(np.clip(abs(normal[2]) / np.linalg.norm(normal), 0, 1)))
triangle_mean_z = float(np.mean(triangle[:, 2]))
```

`measure_polygon_3d` uses 3D-area weights for slope and projected-area weights for mean Z. `ground_elevation` is the median of all GroundSurface exterior Z values. `g1_height` is the projected-area-weighted roof mean Z minus ground elevation. `roof_relief_ratio` is `(max_roof_z - min_roof_z) / (max_roof_z - ground_z)` and returns `NaN` for a non-positive denominator.

- [ ] **Step 4: Run geometry tests**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_roof_geometry -v`

Expected: 3 tests PASS with numeric tolerance `1e-6`.

- [ ] **Step 5: Commit the geometry kernel**

```bash
git add casebase/lod2_roof_retrieval_experiment/roof_geometry.py casebase/lod2_roof_retrieval_experiment/tests/test_roof_geometry.py
git commit -m "feat: add roof geometry metrics"
```

### Task 2: Isolated database adapter and experiment tables

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/database.py`
- Create: `casebase/lod2_roof_retrieval_experiment/tests/test_database.py`

**Interfaces:**
- Consumes: project `.env`, a psycopg2 connection, and Luxembourg source relations.
- Produces: `connect_from_env`, `fetch_lod2_metric_srid`, `iter_luxembourg_buildings`, `fetch_luxembourg_blocks`, `ensure_experiment_schema`, `replace_building_metrics`, `replace_block_metrics`, and `replace_retrieval_results`.

- [ ] **Step 1: Write a failing schema-safety test**

Use fake connection/cursor objects that capture SQL. Assert that `ensure_experiment_schema` contains `CREATE SCHEMA IF NOT EXISTS exp_lod2_roof_lu`, creates the three named experiment tables, and contains no write statement targeting `block.`, `lod1.`, `lod2.`, or `summary.`. Assert that `assert_experiment_relation("lod2.foo")` raises `ValueError`.

- [ ] **Step 2: Run the database test and verify failure**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_database -v`

Expected: FAIL because `database` does not exist.

- [ ] **Step 3: Implement read streaming and write allowlisting**

Use constants:

```python
EXPERIMENT_SCHEMA = "exp_lod2_roof_lu"
BUILDING_TABLE = f"{EXPERIMENT_SCHEMA}.building_metrics"
BLOCK_TABLE = f"{EXPERIMENT_SCHEMA}.block_metrics"
RETRIEVAL_TABLE = f"{EXPERIMENT_SCHEMA}.retrieval_results"
```

`iter_luxembourg_buildings` must use a named server-side cursor, `itersize=2000`, and one ordered query joining matched buildings to RoofSurface/GroundSurface rows. Geometry is returned as WKB after `ST_Transform(geom_3d, lod2_metric_srid)`; footprints are returned as WKB in the same CRS. Group consecutive rows by `building_id` so callers receive one building and its complete roof/ground polygon lists without loading every surface at once.

`ensure_experiment_schema` creates the schema and tables. Each replacement function validates its target with `assert_experiment_relation`, truncates only its experiment table, and inserts rows using `psycopg2.extras.execute_values` in batches of 1000.

- [ ] **Step 4: Run database unit tests**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_database -v`

Expected: all database safety tests PASS without connecting to PostgreSQL.

- [ ] **Step 5: Run a read-only Luxembourg source smoke check**

Open a connection, set `readonly=True`, fetch `lod2_metric_srid`, one block, and one grouped building record. Assert the SRID equals 2169, the block ID begins `LU_LU_`, and the building has at least one roof and one ground polygon. Do not call any schema/table creation function in this check.

- [ ] **Step 6: Commit the database adapter**

```bash
git add casebase/lod2_roof_retrieval_experiment/database.py casebase/lod2_roof_retrieval_experiment/tests/test_database.py
git commit -m "feat: isolate roof experiment database access"
```

### Task 3: Building and block metric aggregation

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/block_metrics.py`
- Create: `casebase/lod2_roof_retrieval_experiment/tests/test_block_metrics.py`

**Interfaces:**
- Consumes: grouped building records from `database.iter_luxembourg_buildings` and block geometry records from `database.fetch_luxembourg_blocks`.
- Produces: `compute_building_metric_row`, `compute_building_metrics`, `compute_block_metrics`, `select_final_m1_features`, and quality reason codes.

- [ ] **Step 1: Write failing aggregation tests**

Construct two synthetic building-metric rows in one 100 m × 100 m block. Give both buildings footprint area 100 m², G1 height 10 m, and equal 3D roof area; give their mean roof slopes 0° and 45°. Assert:

```python
self.assertAlmostEqual(block_row["bcr"], 0.02)
self.assertAlmostEqual(block_row["volume_density"], 0.02 * 10.0)
self.assertEqual(block_row["building_count"], 2)
self.assertAlmostEqual(block_row["roof_mean_slope"], 22.5)
self.assertAlmostEqual(block_row["building_roof_slope_sd"], 22.5)
```

Add cases asserting that a building without roof data receives `missing_roof`, a block below 80% usable footprint coverage receives `coverage_below_80pct`, and a single-building block has roof-slope SD equal to 0.

- [ ] **Step 2: Run aggregation tests and verify failure**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_block_metrics -v`

Expected: FAIL because `block_metrics` does not exist.

- [ ] **Step 3: Implement building metrics**

For every matched building compute footprint area/centroid/orientation, G1 height, roof mean slope, roof relief ratio, roof and ground surface counts, valid roof triangle count, `is_usable`, and semicolon-separated warning/reason codes. Multiple surfaces are accepted. Failed surfaces are skipped and counted.

- [ ] **Step 4: Implement the 12 M1 candidates and three M2 metrics**

Aggregate with NumPy/Pandas. Use population standard deviation (`ddof=0`), CV only when the mean is positive, axial orientation entropy with 18 bins over `[0, 180)`, and nearest-neighbour centroid distance normalized by `sqrt(block_area)`. Compute overall `roof_mean_slope` by 3D roof-area weights, the median building relief ratio, and population SD of building mean slopes.

- [ ] **Step 5: Implement deterministic correlation pruning**

`select_final_m1_features(frame, threshold=0.90)` starts from the ordered 12-feature list. For each pair above the threshold, drop the feature with more missing values; on a tie, drop the later feature in the declared list. Return both the retained list and a decision table containing the pair, correlation, dropped feature, and reason.

- [ ] **Step 6: Run aggregation tests**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_block_metrics -v`

Expected: all aggregation and quality-rule tests PASS.

- [ ] **Step 7: Commit metric aggregation**

```bash
git add casebase/lod2_roof_retrieval_experiment/block_metrics.py casebase/lod2_roof_retrieval_experiment/tests/test_block_metrics.py
git commit -m "feat: aggregate Luxembourg block metrics"
```

### Task 4: Deterministic development-query retrieval

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/retrieval.py`
- Create: `casebase/lod2_roof_retrieval_experiment/tests/test_retrieval.py`

**Interfaces:**
- Consumes: usable `block_metrics` frame, final M1 feature names, roof feature names, `pool_size`, `top_k`, and lambda values.
- Produces: `RobustFeatureScaler`, `select_development_queries`, `retrieve_for_query`, and `run_development_retrieval`.

- [ ] **Step 1: Write failing retrieval tests**

Build a six-row frame with known general and roof coordinates. Assert median/IQR scaling, self-exclusion, a shared M1 candidate pool, Top-K size, stable tie-breaking by `block_id`, and that increasing lambda can select a roof-closer candidate without introducing an ID outside the general pool.

- [ ] **Step 2: Run retrieval tests and verify failure**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_retrieval -v`

Expected: FAIL because `retrieval` does not exist.

- [ ] **Step 3: Implement robust group distances**

Fit median and IQR on the full usable candidate table; replace zero IQR with 1. Compute each group distance as root mean squared standardized feature difference so group weight does not depend on feature count:

```python
d_general = np.sqrt(np.mean((general_z - query_general_z) ** 2, axis=1))
d_roof = np.sqrt(np.mean((roof_z - query_roof_z) ** 2, axis=1))
d_combined = (1.0 - lambda_value) * d_general + lambda_value * d_roof
```

Use the nearest 50 by `(d_general, block_id)` as the shared pool. M1 is the first five by general distance. M2 is the first five by `(d_combined, block_id)` for each lambda.

- [ ] **Step 4: Implement five deterministic development queries**

Use roof-feature quantiles to choose the closest block to these fixed targets: lowest mean slope, highest mean slope, median mean slope with low dispersion, median mean slope with high dispersion, and highest relief ratio. Prevent duplicate query IDs by choosing the next nearest unused block.

- [ ] **Step 5: Run retrieval tests**

Run: `conda run --no-capture-output -n urbanCode python -m unittest tests.test_retrieval -v`

Expected: all retrieval tests PASS.

- [ ] **Step 6: Commit retrieval**

```bash
git add casebase/lod2_roof_retrieval_experiment/retrieval.py casebase/lod2_roof_retrieval_experiment/tests/test_retrieval.py
git commit -m "feat: add paired roof retrieval"
```

### Task 5: Notebook orchestration and saved outputs

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/01_luxembourg_metrics_and_retrieval.ipynb`
- Create: `casebase/lod2_roof_retrieval_experiment/outputs/.gitkeep`
- Create: `casebase/lod2_roof_retrieval_experiment/outputs/.gitignore`
- Modify: `casebase/lod2_roof_retrieval_experiment/EXPERIMENT_DESIGN.md`

**Interfaces:**
- Consumes: all module APIs from Tasks 1–4 and `.env` at the repository root.
- Produces: the three experiment tables plus the CSV/Parquet/JSON files specified by the design.

- [ ] **Step 1: Create a clean notebook with the bundled notebook scaffold helper**

Create the notebook with these sections: purpose/safety, imports/config, database connection and source summary, schema setup, building metric computation, block aggregation, quality summary, correlation pruning, file/database persistence, development-query selection, and retrieval results. Set `RANDOM_SEED = 20260908`, `EXPERIMENT_SCHEMA = "exp_lod2_roof_lu"`, `POOL_SIZE = 50`, `TOP_K = 5`, and `LAMBDA_VALUES = [0.20, 0.35, 0.50]` in one configuration cell.

- [ ] **Step 2: Add a database-write safety switch**

The notebook default is `WRITE_EXPERIMENT_SCHEMA = True`. Before any write, display the exact schema/table targets and require `database.assert_experiment_relation` for every target. Reads always target existing source tables; no source-table mutation SQL appears in notebook cells.

- [ ] **Step 3: Save auditable outputs**

Write `quality_flags.csv`, `building_metrics.parquet`, `block_metrics.parquet`, `experiment_config.json`, `development_queries.csv`, and `development_retrieval_results.csv`. The config records SRID, quality threshold, features before/after pruning, correlation decisions, scaling medians/IQRs, pool size, Top-K, lambdas, package versions, and execution timestamp.

- [ ] **Step 4: Keep generated artifacts out of Git**

Use:

```gitignore
*
!.gitignore
!.gitkeep
```

in `outputs/.gitignore`.

- [ ] **Step 5: Validate notebook structure without executing database writes**

Run a small `nbformat` check asserting that the notebook parses, contains at least ten code/markdown cells, has no outputs, has no execution counts, and contains no `UPDATE`, `DELETE`, `DROP`, or `ALTER` statement targeting source schemas.

- [ ] **Step 6: Run the full unit suite**

Run: `conda run --no-capture-output -n urbanCode python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit notebook orchestration**

```bash
git add casebase/lod2_roof_retrieval_experiment
git commit -m "feat: add Luxembourg roof retrieval notebook"
```

### Task 6: End-to-end verification and handoff

**Files:**
- Modify only if verification exposes a defect: files created in Tasks 1–5.

**Interfaces:**
- Consumes: the completed experiment package and the user's existing PostgreSQL database.
- Produces: evidence that imports, pure tests, read-only source access, and notebook structure are correct; database writes remain scoped to the experiment schema.

- [ ] **Step 1: Run syntax and import verification**

Run `python -m compileall` on the experiment folder and import `roof_geometry`, `database`, `block_metrics`, and `retrieval` from the notebook directory.

- [ ] **Step 2: Run all unit tests again**

Run: `conda run --no-capture-output -n urbanCode python -m unittest discover -s tests -v`

Expected: all tests PASS with no network access and no database writes.

- [ ] **Step 3: Run a read-only 10-building integration sample**

Fetch ten grouped Luxembourg buildings, compute their metrics, and assert that all usable rows have positive G1 height, non-negative 3D roof area, slope in `[0, 90]`, and relief ratio in `[0, 1]`. Fetch their blocks and confirm all IDs join.

- [ ] **Step 4: Inspect repository changes**

Confirm that only the experiment folder and plan/spec files changed, no generated data is staged, and no existing pipeline file was modified.

- [ ] **Step 5: Commit any verification-only fixes**

If verification required fixes, commit only those files with `fix: verify Luxembourg roof experiment`. If no fixes were required, do not create an empty commit.
