"""Bundled reference configs must stay byte-identical to the repo's configs/.

`ac/packaged_configs/` ships the three reference base-model configs inside
the wheel so that README quickstart paths (e.g. `configs/mistral_7b.json`)
resolve in a clean PyPI install via `baseline._resolve_baseline_path`.
The copies must never diverge from the frozen top-level `configs/` files —
this test is the sync guard.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from ac.baseline import PACKAGED_REFERENCE_CONFIGS


class PackagedConfigsSyncTests(unittest.TestCase):
    def test_packaged_listing_matches_directory(self):
        packaged_dir = ROOT / "ac" / "packaged_configs"
        on_disk = sorted(p.name for p in packaged_dir.glob("*.json"))
        self.assertEqual(sorted(PACKAGED_REFERENCE_CONFIGS), on_disk)

    def test_bundled_copies_are_byte_identical_to_configs(self):
        for name in PACKAGED_REFERENCE_CONFIGS:
            canonical = (ROOT / "configs" / name).read_bytes()
            bundled = (ROOT / "ac" / "packaged_configs" / name).read_bytes()
            self.assertEqual(
                canonical, bundled,
                f"ac/packaged_configs/{name} diverged from configs/{name}. "
                "configs/ is the frozen source of truth: re-copy the file "
                "(do not edit the bundled copy in place).",
            )

    def test_missing_path_falls_back_to_bundled_copy(self):
        from ac.baseline import _resolve_baseline_path

        resolved = _resolve_baseline_path("configs/mistral_7b.json")
        # Repo checkout: the local file exists, so no fallback.
        self.assertEqual(resolved, "configs/mistral_7b.json")

        fallback = _resolve_baseline_path("/nonexistent/configs/mistral_7b.json")
        self.assertTrue(fallback.endswith("packaged_configs/mistral_7b.json"))
        self.assertTrue(Path(fallback).is_file())

    def test_unknown_missing_path_is_not_intercepted(self):
        from ac.baseline import _resolve_baseline_path

        bogus = "/nonexistent/configs/my_private_model.json"
        self.assertEqual(_resolve_baseline_path(bogus), bogus)


if __name__ == "__main__":
    unittest.main()
