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
