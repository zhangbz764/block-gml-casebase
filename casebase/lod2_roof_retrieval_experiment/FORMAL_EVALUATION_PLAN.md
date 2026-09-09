# Luxembourg Formal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the Luxembourg development method, create 20 non-development queries, evaluate retrieval without human scores, and establish method rationale plus an honest early conference abstract.

**Architecture:** A new pure-Python `evaluation.py` selects a fixed formal query set and calculates distribution-based roof similarity from the existing building metrics. A second output-free notebook consumes the first notebook's Parquet/JSON outputs and writes only two new tables inside `exp_lod2_roof_lu`. Method rationale and the abstract draft remain separate Markdown artifacts so the notebook stays executable and concise.

**Tech Stack:** Python 3; NumPy, Pandas, SciPy, scikit-learn, nbformat, psycopg2; PostgreSQL/PostGIS; Jupyter Notebook; Markdown.

**Spec:** `casebase/lod2_roof_retrieval_experiment/EXPERIMENT_DESIGN.md`

## Global Constraints

- Luxembourg is the development city, not the final geographic scope.
- Final validation must cover multiple cities and thousands to tens of thousands of usable blocks using parameters frozen before those cities are evaluated.
- Human or expert scores are not part of the current evaluation.
- The five development queries are excluded from the 20 formal queries.
- The fixed primary configuration is M1's retained 11 features, roof weight `lambda=0.35`, general pool size 50, and Top-5 output.
- Existing source schemas remain read-only; database writes remain inside `exp_lod2_roof_lu`.
- Existing notebook `01_luxembourg_metrics_and_retrieval.ipynb` is not modified because it currently contains user-generated run outputs.

---

### Task 1: Method rationale and early abstract

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/METHOD_RATIONALE.md`
- Create: `casebase/lod2_roof_retrieval_experiment/ABSTRACT_DRAFT.md`
- Modify: `casebase/lod2_roof_retrieval_experiment/EXPERIMENT_DESIGN.md`

**Interfaces:**
- Consumes: current Luxembourg results and primary literature/official documentation.
- Produces: the authoritative parameter rationale, multi-city validation position, and an approximately 1000-Chinese-character draft that labels unfinished multi-city quantities explicitly.

- [ ] **Step 1: Record sources and evidence roles**

Use official CityGML documentation for LOD semantics, the GBMI paper and multi-city morphology research for M1 feature families, official scikit-learn documentation for median/IQR scaling, and official SciPy documentation for Wasserstein distance. Distinguish literature-backed design choices from development-set choices such as `lambda=0.35`.

- [ ] **Step 2: Write `METHOD_RATIONALE.md`**

For every feature group and parameter, record: measured concept, formula or rule, reason for inclusion, basis type (`disciplinary`, `literature`, `development sensitivity`, or `practical retrieval`), and limitation. State that the exact weight is not a universal constant.

- [ ] **Step 3: Write `ABSTRACT_DRAFT.md`**

Write background, objective, data, method, current Luxembourg result, planned multi-city validation, conclusion boundary, and two-figure plan. Use explicit `【多城市实验后替换】` markers wherever final sample counts or multi-city effects are not yet known.

- [ ] **Step 4: Update the experiment design**

Describe Luxembourg as the development city; require multi-city external validation with fixed definitions/parameters and a final scale of thousands to tens of thousands of usable blocks.

- [ ] **Step 5: Commit documentation**

```text
git add casebase/lod2_roof_retrieval_experiment/METHOD_RATIONALE.md casebase/lod2_roof_retrieval_experiment/ABSTRACT_DRAFT.md casebase/lod2_roof_retrieval_experiment/EXPERIMENT_DESIGN.md casebase/lod2_roof_retrieval_experiment/FORMAL_EVALUATION_PLAN.md
git commit -m "docs: frame formal and multicity roof evaluation"
```

### Task 2: Formal query selection

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/evaluation.py`
- Create: `casebase/lod2_roof_retrieval_experiment/tests/test_evaluation.py`

**Interfaces:**
- Consumes: usable block metrics and five development query IDs.
- Produces: `select_formal_queries(blocks, development_ids, seed=20260908, regular_count=16) -> pd.DataFrame`.

- [ ] **Step 1: Write failing selection tests**

Create at least 40 synthetic blocks spanning four roof-slope quartiles. Assert 20 unique queries, no overlap with development IDs, four regular queries per quartile, and four unique boundary queries named `near_flat`, `steep`, `high_relief`, and `high_dispersion`.

- [ ] **Step 2: Run the focused test and confirm failure**

Run `conda run --no-capture-output -n urbanCode python -m unittest tests.test_evaluation.FormalQueryTests -v` and expect failure because `evaluation.py` does not exist.

- [ ] **Step 3: Implement deterministic selection**

Exclude development IDs. Select the four boundary cases first using stable `(metric, block_id)` ordering. Remove them, divide remaining blocks into four slope quartiles with `pd.qcut`, and select four per quartile with `DataFrame.sample(random_state=seed + quartile_index)`. Return query role, stratum, and the three roof features.

- [ ] **Step 4: Run focused tests**

Run the same unittest target and expect all selection tests to pass.

### Task 3: Distribution-based non-scoring evaluation

**Files:**
- Modify: `casebase/lod2_roof_retrieval_experiment/evaluation.py`
- Modify: `casebase/lod2_roof_retrieval_experiment/tests/test_evaluation.py`

**Interfaces:**
- Consumes: building metrics and M1/M2 retrieval rows.
- Produces: `evaluate_roof_distributions(buildings, retrieval) -> pd.DataFrame` and `summarize_formal_evaluation(enriched) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write failing distribution tests**

Use literal synthetic building slopes/reliefs for one query and two candidates. Assert Wasserstein distances in degrees and unitless relief, absolute flat-roof-share difference using slope `<5°`, and query-level Top-5 means. Assert missing query/candidate building data raises `ValueError` rather than silently returning zero.

- [ ] **Step 2: Run focused tests and confirm expected failure**

Run `conda run --no-capture-output -n urbanCode python -m unittest tests.test_evaluation.RoofDistributionTests -v`.

- [ ] **Step 3: Implement the minimal evaluator**

Group usable building rows by block ID. Use `scipy.stats.wasserstein_distance` for building mean-roof-slope and relief distributions. Calculate flat-roof share as the proportion of buildings with mean slope below 5°. Append the three held-out evaluation fields to each retrieval row.

- [ ] **Step 4: Implement summaries**

Produce query-level means by method/lambda and an aggregate table containing mean, median, and paired M2-minus-M1 change for general distance plus the three distribution measures. Do not convert these measures into a composite score.

- [ ] **Step 5: Run all evaluation tests**

Run `conda run --no-capture-output -n urbanCode python -m unittest tests.test_evaluation -v` and expect all tests to pass.

- [ ] **Step 6: Commit evaluation logic**

```text
git add casebase/lod2_roof_retrieval_experiment/evaluation.py casebase/lod2_roof_retrieval_experiment/tests/test_evaluation.py
git commit -m "feat: add non-scoring roof evaluation"
```

### Task 4: Isolated formal-evaluation tables

**Files:**
- Modify: `casebase/lod2_roof_retrieval_experiment/database.py`
- Modify: `casebase/lod2_roof_retrieval_experiment/tests/test_database.py`

**Interfaces:**
- Produces: `FORMAL_QUERY_TABLE`, `FORMAL_EVALUATION_TABLE`, updated write allowlist, table creation, and replacement helpers.

- [ ] **Step 1: Write failing database safety tests**

Assert both new relations are under `exp_lod2_roof_lu`, are created by `ensure_experiment_schema`, are accepted by the allowlist, and source-schema relations remain rejected.

- [ ] **Step 2: Run database tests and confirm failure**

Run `conda run --no-capture-output -n urbanCode python -m unittest tests.test_database -v`.

- [ ] **Step 3: Add two tables and replacement helpers**

Create `formal_queries` for the fixed query set and `formal_evaluation_results` for retrieval plus distribution metrics. Use the existing experiment-only replacement boundary and batch insertion pattern.

- [ ] **Step 4: Run database tests**

Run the same unittest target and expect all tests to pass without a database connection.

- [ ] **Step 5: Commit database extension**

```text
git add casebase/lod2_roof_retrieval_experiment/database.py casebase/lod2_roof_retrieval_experiment/tests/test_database.py
git commit -m "feat: store formal roof evaluation separately"
```

### Task 5: Formal Luxembourg notebook

**Files:**
- Create: `casebase/lod2_roof_retrieval_experiment/02_luxembourg_formal_evaluation.ipynb`
- Modify: `casebase/lod2_roof_retrieval_experiment/tests/test_notebook.py`

**Interfaces:**
- Consumes: `outputs/block_metrics.parquet`, `outputs/building_metrics.parquet`, `outputs/development_queries.csv`, and `outputs/experiment_config.json`.
- Produces: `outputs/formal_luxembourg/formal_queries.csv`, `formal_retrieval_results.csv`, `formal_evaluation_results.csv`, `query_summary.csv`, and `aggregate_summary.csv`, plus the two formal database tables when writes are enabled.

- [ ] **Step 1: Scaffold the notebook**

Use the bundled notebook helper with kind `experiment` and title `Luxembourg Formal Roof Retrieval Evaluation`.

- [ ] **Step 2: Add focused cells**

Include purpose, frozen configuration, input checks, formal query selection, fixed-lambda retrieval, distribution evaluation, aggregate comparison, isolated persistence, and interpretation boundaries. Read the 11 retained M1 features from `experiment_config.json`; assert `lambda=0.35`, pool 50, and Top-5 before running.

- [ ] **Step 3: Add notebook tests**

Assert the new notebook parses, has no outputs or execution counts, contains all expected output names, fixes lambda to 0.35, and writes only through database allowlisted helpers.

- [ ] **Step 4: Run all tests**

Run `conda run --no-capture-output -n urbanCode python -m unittest discover -s tests -v`.

- [ ] **Step 5: Execute the notebook top-to-bottom**

Use the `urbanCode` kernel with the existing project `.env`. Confirm 20 unique formal queries, 200 retrieval rows (20 queries × 5 M1 results + 20 queries × 5 M2 results), finite distribution metrics, both database tables written only under `exp_lod2_roof_lu`, and all five output files created.

- [ ] **Step 6: Clear committed notebook outputs**

After recording the numeric results externally, clear outputs and execution counts from the committed notebook. Leave generated files ignored under `outputs/`.

- [ ] **Step 7: Commit orchestration**

```text
git add casebase/lod2_roof_retrieval_experiment/02_luxembourg_formal_evaluation.ipynb casebase/lod2_roof_retrieval_experiment/tests/test_notebook.py
git commit -m "feat: add formal Luxembourg roof evaluation"
```

### Task 6: Verification and next-stage handoff

**Files:**
- Modify only if a defect is found in Tasks 1–5.

- [ ] **Step 1: Run the full unittest suite**

Run `conda run --no-capture-output -n urbanCode python -m unittest discover -s tests -v` and require zero failures.

- [ ] **Step 2: Compile all experiment modules**

Run `conda run --no-capture-output -n urbanCode python -m compileall -q casebase/lod2_roof_retrieval_experiment`.

- [ ] **Step 3: Inspect Git scope**

Confirm the user's modified `01_luxembourg_metrics_and_retrieval.ipynb` remains unstaged and is not included in any new commit. Confirm generated outputs remain ignored.

- [ ] **Step 4: Report formal results and multi-city readiness**

Report the 20-query results without claiming that geometric distribution distance equals design quality. Identify which code boundaries must next be parameterized by city and recommend multiple validation cities based on available LOD1/LOD2 coverage.
