"""Pinned download integrity and v10 dependency staging failures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_optiscaler import fetch
from stage_optiscaler import stage, REQUIRED


class OptiScalerReleaseTests(unittest.TestCase):
    def test_corrupt_cached_archive_is_rejected_without_downloading(self):
        lock = json.loads((ROOT / 'packaging/optiscaler-release.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / lock['tag'] / lock['url'].rsplit('/', 1)[1]
            path.parent.mkdir()
            path.write_bytes(b'corrupt archive')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                fetch(temporary)
            self.assertEqual(path.read_bytes(), b'corrupt archive')

    def test_missing_fg_provider_is_rejected_before_staging(self):
        for missing in REQUIRED:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                source, output = Path(temporary) / 'source', Path(temporary) / 'output'
                for name in ('OptiScaler.dll', 'OptiScaler.ini', 'Licenses/XeSS_LICENSE.txt',
                             'Licenses/FidelityFX_v2_LICENSE.md',
                             *('OptiScaler/' + name for name in REQUIRED if name != missing)):
                    path = source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'fixture')
                with self.assertRaisesRegex(ValueError, missing):
                    stage(source, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
