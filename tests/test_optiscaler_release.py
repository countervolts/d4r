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

    def test_patched_build_replaces_dll_and_ini(self):
        windows = ROOT / 'packaging/optiscaler-windows.settings'
        sections = {}
        for settings in (ROOT / 'packaging/optiscaler.settings', windows):
            for line in settings.read_text().splitlines():
                if line and not line.startswith('#'):
                    section, key = line.split('=', 1)[0].split('.', 1)
                    sections.setdefault(section, set()).add(key)
        ini = ''.join(f'[{s}]\r\n' + ''.join(f'{k}=auto\r\n' for k in sorted(keys)) for s, keys in sections.items())
        with tempfile.TemporaryDirectory() as temporary:
            source, built, output = (Path(temporary) / name for name in ('source', 'built', 'output'))
            for name in ('OptiScaler.dll', 'Licenses/XeSS_LICENSE.txt', 'Licenses/FidelityFX_v2_LICENSE.md',
                         *('OptiScaler/' + name for name in REQUIRED)):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'release')
            # The release's INI lacks the key the patch adds, so only the patched INI can take the settings.
            (source / 'OptiScaler.ini').write_bytes(ini.replace('ForceNvidia=auto\r\n', '').encode())
            built.mkdir()
            (built / 'OptiScaler.dll').write_bytes(b'patched')
            (built / 'OptiScaler.ini').write_bytes(ini.encode())
            (built / '0001-fixture.patch').write_bytes(b'patch')
            with self.assertRaisesRegex(ValueError, 'ForceNvidia'):
                stage(source, output, windows)
            stage(source, output, windows, patched=built)
            self.assertEqual((output / 'dxgi.dll').read_bytes(), b'patched')
            staged = (output / 'OptiScaler.ini').read_bytes()
            self.assertIn(b'ForceNvidia=true\r\n', staged)
            self.assertIn(b'UseFakenvapi=true\r\n', staged)
            provenance = json.loads((output / 'd4r/source/optiscaler.json').read_text())
            self.assertEqual(list(provenance['patched']['patches']), ['0001-fixture.patch'])
            self.assertEqual(provenance['release'], {'version': 'custom v10 input'})


if __name__ == '__main__':
    unittest.main()
