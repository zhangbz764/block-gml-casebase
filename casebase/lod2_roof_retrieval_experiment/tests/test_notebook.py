import re
import unittest
from pathlib import Path

import nbformat


class NotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "01_luxembourg_metrics_and_retrieval.ipynb"
        cls.notebook = nbformat.read(path, as_version=4)

    def test_notebook_is_substantial_and_output_free(self):
        self.assertGreaterEqual(len(self.notebook.cells), 10)
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

    def test_every_code_cell_has_valid_python_syntax(self):
        for index, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"notebook-cell-{index}", "exec")

    def test_configuration_and_safety_switch_are_explicit(self):
        source = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertIn('EXPERIMENT_SCHEMA = "exp_lod2_roof_lu"', source)
        self.assertIn("WRITE_EXPERIMENT_SCHEMA = True", source)
        self.assertIn("TARGET_BLOCK_COUNT = 500", source)
        self.assertIn("db.assert_experiment_relation", source)

    def test_no_source_schema_mutation_sql(self):
        source = "\n".join(cell.source for cell in self.notebook.cells)
        forbidden = re.compile(
            r"(?:UPDATE|DELETE\s+FROM|DROP\s+TABLE|ALTER\s+TABLE|TRUNCATE\s+TABLE)"
            r"\s+(?:block|lod1|lod2|summary)\.",
            re.IGNORECASE,
        )
        self.assertIsNone(forbidden.search(source))


if __name__ == "__main__":
    unittest.main()
