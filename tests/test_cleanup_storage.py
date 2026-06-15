import unittest
from pathlib import Path

from cleanup_storage import DATA_ROOT, PROTECTED_DATA_ROOT, is_under


class CleanupStorageTests(unittest.TestCase):
    def test_data_jiang_paths_are_detected_as_protected(self) -> None:
        root = Path("/data/jiang")
        self.assertTrue(is_under(Path("/data/jiang/vennemdp/hf_models"), root))
        self.assertTrue(is_under(Path("/data/jiang/vennemdp/audit"), root))
        self.assertFalse(is_under(Path("/home/vennemdp/.cache/huggingface"), root))

    def test_include_data_root_should_not_authorize_other_data_paths(self) -> None:
        legacy = Path("/data/jiang/vennemdp/hf_models")
        self.assertTrue(is_under(legacy, PROTECTED_DATA_ROOT))
        self.assertNotEqual(legacy, DATA_ROOT.resolve())


if __name__ == "__main__":
    unittest.main()
