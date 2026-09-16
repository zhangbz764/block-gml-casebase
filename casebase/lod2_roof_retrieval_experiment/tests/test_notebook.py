import ast
import importlib.metadata as metadata
import re
import unittest
from pathlib import Path

import nbformat


class NotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "01_luxembourg_metrics_and_retrieval.ipynb"
        cls.notebook = nbformat.read(path, as_version=4)
        cls.formal_path = (
            Path(__file__).parents[1] / "02_luxembourg_formal_evaluation.ipynb"
        )
        cls.multicity_path = (
            Path(__file__).parents[1] / "03_multicity_pool_and_retrieval.ipynb"
        )
        cls.multicity_visual_path = (
            Path(__file__).parents[1] / "04_multicity_visual_diagnostics.ipynb"
        )
        cls.global_query_path = (
            Path(__file__).parents[1] / "05_global_query_selection_and_retrieval.ipynb"
        )
        cls.figure2_path = (
            Path(__file__).parents[1] / "06_figure2_panel_export.ipynb"
        )

    def test_metrics_notebook_is_substantial(self):
        self.assertGreaterEqual(len(self.notebook.cells), 10)

    def test_formal_notebook_contains_fixed_flow_and_visualization(self):
        self.assertTrue(self.formal_path.is_file())
        notebook = nbformat.read(self.formal_path, as_version=4)
        self.assertGreaterEqual(len(notebook.cells), 10)
        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("PRIMARY_LAMBDA = 0.35", source)
        self.assertIn("FORMAL_QUERY_COUNT = 20", source)
        self.assertIn("WRITE_EXPERIMENT_SCHEMA = True", source)
        self.assertIn("replace_formal_evaluation_results", source)
        self.assertIn("formal_luxembourg' / 'figures", source)
        self.assertIn("QUERY_IDS_TO_DRAW", source)
        self.assertIn("save_query_plan_comparison", source)
        self.assertIn("save_query_axonometric_comparison", source)
        self.assertIn("formal_evaluation_overview.png", source)
        self.assertIn("_plan.png", source)
        self.assertIn("_axonometric.png", source)

    def test_every_code_cell_has_valid_python_syntax(self):
        notebooks = [self.notebook]
        if self.formal_path.is_file():
            notebooks.append(nbformat.read(self.formal_path, as_version=4))
        if self.multicity_path.is_file():
            notebooks.append(nbformat.read(self.multicity_path, as_version=4))
        if self.multicity_visual_path.is_file():
            notebooks.append(nbformat.read(self.multicity_visual_path, as_version=4))
        if self.global_query_path.is_file():
            notebooks.append(nbformat.read(self.global_query_path, as_version=4))
        for notebook_index, notebook in enumerate(notebooks):
            for cell_index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(
                        cell.source,
                        f"notebook-{notebook_index}-cell-{cell_index}",
                        "exec",
                    )

    def test_configuration_and_safety_switch_are_explicit(self):
        source = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertIn('EXPERIMENT_SCHEMA = "exp_lod2_roof_lu"', source)
        self.assertIn("WRITE_EXPERIMENT_SCHEMA = True", source)
        self.assertIn("TARGET_BLOCK_COUNT = 500", source)
        self.assertIn("db.assert_experiment_relation", source)

    def test_no_source_schema_mutation_sql(self):
        notebooks = [self.notebook]
        if self.formal_path.is_file():
            notebooks.append(nbformat.read(self.formal_path, as_version=4))
        if self.multicity_path.is_file():
            notebooks.append(nbformat.read(self.multicity_path, as_version=4))
        if self.multicity_visual_path.is_file():
            notebooks.append(nbformat.read(self.multicity_visual_path, as_version=4))
        if self.global_query_path.is_file():
            notebooks.append(nbformat.read(self.global_query_path, as_version=4))
        source = "\n".join(
            cell.source for notebook in notebooks for cell in notebook.cells
        )
        forbidden = re.compile(
            r"(?:UPDATE|DELETE\s+FROM|DROP\s+TABLE|ALTER\s+TABLE|TRUNCATE\s+TABLE)"
            r"\s+(?:block|lod1|lod2|summary)\.",
            re.IGNORECASE,
        )
        self.assertIsNone(forbidden.search(source))

    def test_multicity_notebook_is_clean_and_uses_frozen_cross_city_flow(self):
        self.assertTrue(self.multicity_path.is_file())
        notebook = nbformat.read(self.multicity_path, as_version=4)
        self.assertGreaterEqual(len(notebook.cells), 14)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

        source = "\n".join(cell.source for cell in notebook.cells)
        tree = ast.parse(
            "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        )
        literal_assignments = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    literal_assignments[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

        self.assertEqual(literal_assignments["EXPERIMENT_SCHEMA"], "exp_lod2_roof_multi")
        self.assertEqual(literal_assignments["TARGET_PER_CITY"], 5000)
        self.assertEqual(literal_assignments["BATCH_SIZE"], 1000)
        self.assertEqual(literal_assignments["BASE_SEED"], 20260909)
        self.assertEqual(literal_assignments["MIN_REFERENCE_BCR"], 0.02)
        self.assertEqual(literal_assignments["POOL_SIZE"], 50)
        self.assertEqual(literal_assignments["TOP_K"], 5)
        self.assertEqual(literal_assignments["PRIMARY_LAMBDA"], 0.35)
        self.assertTrue(literal_assignments["WRITE_EXPERIMENT_SCHEMA"])
        self.assertIn("discover_lod2_cities", source)
        self.assertIn("validate_source_columns", source)
        self.assertIn("randomized_block_ids", source)
        self.assertIn("collect_city_sample", source)
        self.assertIn("load_city_checkpoint", source)
        self.assertIn("id_set_fingerprint", source)
        self.assertIn("GLOBAL_POOL_COMPLETE", source)
        self.assertIn("CITIES_TO_RUN is None", source)
        self.assertIn("and not failed_city_rows", source)
        self.assertIn("WRITE_EXPERIMENT_SCHEMA and GLOBAL_POOL_COMPLETE", source)
        self.assertIn("RESULT_OUTPUT_DIR", source)
        self.assertIn("config_fingerprint", source)
        self.assertIn("retrieve_for_query", source)
        self.assertIn("query_city='luxembourg'", source)
        self.assertIn("candidate_city", source)
        self.assertIn("'outputs' / 'multicity'", source)

    def test_multicity_visual_notebook_is_clean_and_reads_saved_results(self):
        self.assertTrue(self.multicity_visual_path.is_file())
        notebook = nbformat.read(self.multicity_visual_path, as_version=4)
        self.assertGreaterEqual(len(notebook.cells), 10)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("global_retrieval_results.csv", source)
        self.assertIn("global_formal_queries.csv", source)
        self.assertIn("QUERY_IDS_TO_RENDER", source)
        self.assertIn("select_category_examples", source)
        self.assertIn("fetch_multicity_scenes", source)
        self.assertIn("save_multicity_plan_comparison", source)
        self.assertIn("save_multicity_axonometric_comparison", source)
        self.assertIn("query_gain_cost.png", source)
        self.assertIn("candidate_city_shift.png", source)
        self.assertNotIn("WRITE_EXPERIMENT_SCHEMA", source)

    def test_global_query_notebook_is_clean_and_builds_frozen_paired_retrieval(self):
        self.assertTrue(self.global_query_path.is_file())
        notebook = nbformat.read(self.global_query_path, as_version=4)
        self.assertGreaterEqual(len(notebook.cells), 12)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

        source = "\n".join(cell.source for cell in notebook.cells)
        tree = ast.parse(
            "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        )
        literal_assignments = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    literal_assignments[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

        self.assertEqual(literal_assignments["QUERIES_PER_CATEGORY"], 10)
        self.assertEqual(literal_assignments["MAX_PER_CITY_PER_CATEGORY"], 3)
        self.assertEqual(literal_assignments["TOP_K"], 5)
        self.assertEqual(literal_assignments["POOL_SIZE"], 50)
        self.assertEqual(literal_assignments["PRIMARY_LAMBDA"], 0.35)
        self.assertIn("build_global_query_set", source)
        self.assertIn("retrieve_for_query", source)
        self.assertIn("global_formal_queries.csv", source)
        self.assertIn("global_retrieval_results.csv", source)
        self.assertNotIn("WRITE_EXPERIMENT_SCHEMA", source)

    def test_ranked_batch_axonometric_cell_is_independent_and_does_not_embed_images(self):
        notebook = nbformat.read(self.multicity_visual_path, as_version=4)
        cells = [cell for cell in notebook.cells if cell.get("id") == "batch-c4"]

        self.assertEqual(len(cells), 1)
        source = cells[0].source
        compile(source, "04-batch-c4", "exec")
        self.assertIn("rank_query_effects", source)
        self.assertIn("all_queries_ranked_axonometric", source)
        self.assertIn("ranked_axonometric_index.csv", source)
        self.assertNotIn("display(Image", source)

    def test_figure2_notebook_exports_four_clean_chinese_panels(self):
        self.assertTrue(self.figure2_path.is_file())
        notebook = nbformat.read(self.figure2_path, as_version=4)
        self.assertGreaterEqual(len(notebook.cells), 8)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                compile(cell.source, "06-figure2-panel-export", "exec")

        source = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("Deng.ttf", source)
        self.assertIn("query_gain_cost_scatter.png", source)
        self.assertIn("13_heterogeneous_DE_BE_002692_axonometric.png", source)
        self.assertIn("15_typical_CZ_PR_000849_axonometric.png", source)
        self.assertIn("16_steep_DE_BE_008010_axonometric.png", source)
        self.assertIn("总体形态检索", source)
        self.assertIn("屋顶增强检索", source)
        self.assertNotIn("display(Image", source)

    def test_every_recorded_package_has_installed_distribution_metadata(self):
        source = "\n".join(
            cell.source for cell in self.notebook.cells if cell.cell_type == "code"
        )
        tree = ast.parse(source)
        package_names = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "package_versions"
                    and isinstance(value, ast.DictComp)
                ):
                    package_names = ast.literal_eval(value.generators[0].iter)
        self.assertIsNotNone(package_names)
        for package_name in package_names:
            try:
                metadata.version(package_name)
            except metadata.PackageNotFoundError:
                self.fail(f"No installed distribution metadata for {package_name}")


if __name__ == "__main__":
    unittest.main()
