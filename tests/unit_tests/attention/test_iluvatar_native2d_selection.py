"""CPU checks for an opt-in hardware-specific attention backend."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/attention/native2d_selection.py"
SPEC = importlib.util.spec_from_file_location("native2d_selection", SOURCE)
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)


class Native2DSelectionTest(unittest.TestCase):
    def test_default_keeps_existing_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(SELECTION.native2d_backend_path("iluvatar"))

    def test_opt_in_selects_plugin_class(self):
        with patch.dict(os.environ, {"VLLM_FL_ILUVATAR_NATIVE2D": "1"}, clear=True):
            self.assertTrue(SELECTION.native2d_backend_path("iluvatar").endswith(
                ".native2d_attention.Native2DAttentionBackend"))

    def test_other_hardware_keeps_existing_backend(self):
        with patch.dict(os.environ, {"VLLM_FL_ILUVATAR_NATIVE2D": "1"}, clear=True):
            for vendor in ("nvidia", "metax", "ascend", "unknown"):
                self.assertIsNone(SELECTION.native2d_backend_path(vendor))

    def test_old_experiments_cannot_be_combined(self):
        for old in ("VLLM_FL_USE_HYBRID_ATTN", "VLLM_FL_USE_FLAGGEMS_ATTN"):
            with patch.dict(os.environ, {"VLLM_FL_ILUVATAR_NATIVE2D": "1", old: "1"}, clear=True):
                with self.assertRaises(ValueError):
                    SELECTION.native2d_backend_path("iluvatar")


if __name__ == "__main__":
    unittest.main()
