import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_paths", ROOT / "src/_release_paths.py")
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class PathTests(unittest.TestCase):
    def test_clone_inputs(self):
        p = paths.release_path("data/starcoder/input/python/starcoderbase_val_data_clone.jsonl")
        self.assertTrue(Path(p).is_file())
        self.assertEqual(Path(p), ROOT / "data/clone/starcoder/input/python/starcoderbase_val_data_clone.jsonl")

    def test_clone_features(self):
        relative = "data/magicoder/input_processed/cpp/Magicoder-CL-7B_top0.1_train_data_clone.pkl"
        self.assertTrue(Path(paths.release_path(relative)).is_file())

    def test_output_separate_from_frozen_results(self):
        p = Path(paths.release_path("data/magicoder/results/cpp/test.txt"))
        self.assertTrue(p.is_relative_to(ROOT / "outputs"))

    def test_output_override(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"CODEMIA_OUTPUT_ROOT": temporary}):
                path = paths.release_path("data/starcoder/results/python/test.txt")
                self.assertEqual(Path(path), Path(temporary) / "starcoder/results/python/test.txt")

    def test_feature_override(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"CODEMIA_FEATURE_ROOT": temporary}):
                path = paths.release_path("data/starcoder/input_processed/python/features.pkl")
                self.assertEqual(Path(path), Path(temporary) / "starcoder/input_processed/python/features.pkl")
