"""Check byte-preserving compaction and refusal to overwrite research changes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_results import MANIFEST, pack, restore


class CompactResultsTests(unittest.TestCase):
    def test_pack_restore_exact_bytes_and_idempotent_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "audit" / "summary.csv"
            source.parent.mkdir()
            content = b"\xef\xbb\xbfid,value\r\nS1,0.123456789012345\r\n"
            source.write_bytes(content)
            archive = root / "records.zip"
            self.assertEqual(pack(root, archive, [source])["packed"], 1)
            self.assertFalse(source.exists())
            self.assertEqual(restore(root, archive), 1)
            self.assertEqual(source.read_bytes(), content)
            self.assertEqual(restore(root, archive), 0)

    def test_changed_file_is_not_overwritten_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manual.csv"
            source.write_bytes(b"original")
            archive = root / "records.zip"
            pack(root, archive, [source])
            source.write_bytes(b"researcher decision")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                restore(root, archive)
            with self.assertRaisesRegex(ValueError, "differs"):
                pack(root, archive, [source])
            self.assertEqual(source.read_bytes(), b"researcher decision")
            self.assertFalse((root / "records.zip.tmp").exists())

    def test_corrupt_archive_refuses_restore_before_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "records.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("audit/summary.csv", b"changed bytes")
                handle.writestr(MANIFEST, json.dumps({"audit/summary.csv": {"bytes": 1, "sha256": "wrong"}}))
            with self.assertRaisesRegex(ValueError, "verification"):
                restore(root, archive)
            self.assertFalse((root / "audit").exists())


if __name__ == "__main__":
    unittest.main()
