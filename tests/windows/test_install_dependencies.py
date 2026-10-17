"""Native installer preservation, rollback and pin contract tests; no downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('install_dependencies', REPO / 'tools/windows/install_dependencies.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def test_download_prompt_and_default_install_for_every_gpu(self):
        targets = json.loads((REPO / 'tools/windows/gpu-targets.json').read_text())['targets']
        (self.runtime / 'd4r.ini').write_text('[RayReconstruction]\nBackend = off\n')
        (self.runtime / 'dll-pins.txt').write_text('')
        for target in targets:
            (self.runtime / 'package.json').write_text(json.dumps({'targets': {target['architecture']: {}}}))
            for answer, proceeds in (('y', True), ('Y', True), ('', True), ('n', False), (' N ', False)):
                with self.subTest(target=target['architecture'], answer=answer), \
                        patch('sys.argv', ['install-dependencies.py', '--package-root', str(self.runtime),
                                           '--cache-root', str(self.root / 'cache'),
                                           '--backup-root', str(self.backups)]), \
                        patch('builtins.input', return_value=answer), \
                        patch('builtins.print'), patch.object(installer, 'PINS', {}), \
                        patch.object(installer, 'collect', return_value=[]) as collect, \
                        patch.object(installer, 'transaction', return_value={}) as transaction, \
                        patch.object(installer, 'absolutize_library_paths'), \
                        patch.object(installer, 'install_model') as model:
                    installer.main()
                    self.assertEqual(collect.called, proceeds)
                    self.assertEqual(transaction.called, proceeds)
                    model.assert_not_called()

    def test_yes_installs_without_prompt_or_model(self):
        (self.runtime / 'd4r.ini').write_text('[RayReconstruction]\nBackend = off\n')
        (self.runtime / 'dll-pins.txt').write_text('')
        with patch('sys.argv', ['install-dependencies.py', '--package-root', str(self.runtime), '--yes',
                                '--cache-root', str(self.root / 'cache'), '--backup-root', str(self.backups)]), \
                patch('builtins.input', side_effect=AssertionError('Unexpected prompt')), \
                patch('builtins.print'), patch.object(installer, 'PINS', {}), \
                patch.object(installer, 'collect', return_value=[]), \
                patch.object(installer, 'transaction', return_value={}) as transaction, \
                patch.object(installer, 'absolutize_library_paths'), \
                patch.object(installer, 'install_model') as model:
            installer.main()
            transaction.assert_called_once()
            model.assert_not_called()

    def test_native_fsr_requires_explicit_rdna4_only_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            self.assertFalse(installer.native_fsr_package(runtime))
            metadata = runtime / 'package.json'
            for targets, wanted in (({}, False), ({'gfx1201': {}}, True),
                                    ({'gfx1200': {}, 'gfx1201': {}}, True),
                                    ({'gfx1101': {}, 'gfx1201': {}}, False)):
                metadata.write_text(json.dumps({'targets': targets}))
                self.assertEqual(installer.native_fsr_package(runtime), wanted)
            (runtime / 'gpu-target.json').write_text(json.dumps({'architecture': 'gfx1101'}))
            self.assertFalse(installer.native_fsr_package(runtime))
            (runtime / 'gpu-target.json').write_text(json.dumps({'architecture': 'gfx1201'}))
            self.assertFalse(installer.native_fsr_package(runtime))  # mixed package still includes RDNA3

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='d4r-installer-test-')
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.runtime = self.root / 'game/d4r'
        self.runtime.mkdir(parents=True)
        self.assets = []
        for index in range(3):
            source = self.root / f'source{index}'
            source.write_bytes(f'new{index}'.encode())
            relative = f'{index}/test.dll'
            target = self.runtime / relative
            target.parent.mkdir()
            target.write_bytes(f'old{index}'.encode())
            self.assets.append((source, relative, installer.sha(source)))
        self.backups = self.root / 'backups'

    def assert_originals(self):
        for index, (_, relative, _) in enumerate(self.assets):
            self.assertEqual((self.runtime / relative).read_bytes(), f'old{index}'.encode())
        self.assertFalse(list(self.runtime.rglob('*.install-*')))
        self.assertFalse(list(self.runtime.rglob('*.rollback-*')))

    def test_backup_idempotence_unrelated(self):
        unrelated = self.runtime / 'd4r.ini'
        unrelated.write_bytes(b'\xef\xbb\xbf; keep\r\n')
        result = installer.transaction(self.runtime, self.assets, self.backups)
        self.assertEqual(result['changed'], 3)
        manifest = json.loads((Path(result['backup']) / 'backup-manifest.json').read_text())
        for index, entry in enumerate(manifest['files']):
            self.assertEqual(Path(entry['backup']).read_bytes(), f'old{index}'.encode())
        self.assertEqual(installer.transaction(self.runtime, self.assets, self.backups)['changed'], 0)
        self.assertEqual(len(list(self.backups.iterdir())), 1)
        self.assertEqual(unrelated.read_bytes(), b'\xef\xbb\xbf; keep\r\n')

    def test_failure_rolls_back_replacements_and_new_files(self):
        (self.runtime / self.assets[0][1]).unlink()
        replace = installer.os.replace
        def fail_third(source, destination):
            if '.install-' in str(source) and destination == self.runtime / self.assets[2][1]:
                raise OSError('injected replacement failure')
            return replace(source, destination)
        with patch.object(installer.os, 'replace', side_effect=fail_third):
            with self.assertRaisesRegex(OSError, 'injected'):
                installer.transaction(self.runtime, self.assets, self.backups)
        self.assertFalse((self.runtime / self.assets[0][1]).exists())
        for index in (1, 2):
            self.assertEqual((self.runtime / self.assets[index][1]).read_bytes(), f'old{index}'.encode())
        self.assertFalse(list(self.runtime.rglob('*.install-*')))

    def test_concurrent_destination_edit_survives_rollback(self):
        replace = installer.os.replace
        def change_second(source, destination):
            result = replace(source, destination)
            if '.install-' in str(source) and destination == self.runtime / self.assets[0][1]:
                (self.runtime / self.assets[1][1]).write_bytes(b'concurrent')
            return result
        with patch.object(installer.os, 'replace', side_effect=change_second):
            with self.assertRaisesRegex(RuntimeError, 'changed after backup'):
                installer.transaction(self.runtime, self.assets, self.backups)
        self.assertEqual((self.runtime / self.assets[0][1]).read_bytes(), b'old0')
        self.assertEqual((self.runtime / self.assets[1][1]).read_bytes(), b'concurrent')

    def test_invalid_staging_never_mutates_runtime(self):
        self.assets[2][0].write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'Staged DLL SHA256'):
            installer.transaction(self.runtime, self.assets, self.backups)
        self.assert_originals()

    def test_pins_match_existing_installer(self):
        original = (REPO / 'scripts/windows/install-dlss.ps1').read_text()
        for _, _, url, digest in installer.PINS.values():
            self.assertIn(digest, original)
            if url:
                self.assertIn(url, original)
        for url, digest in (installer.DRIVER, installer.SEVENZIP):
            self.assertIn(url, original)
            self.assertIn(digest, original)

    def test_fsr_denoiser_pin_matches_setup_script_and_is_optional(self):
        name, relative, url, digest = installer.FSR_DENOISER
        script = (REPO / 'scripts/windows/setup-fsr-rr.ps1').read_text()
        self.assertIn(digest, script)
        self.assertIn(url, script)
        self.assertEqual((name, relative), ('amd_fidelityfx_denoiser_dx12.dll',) * 2)
        self.assertNotIn('fsr', installer.PINS)  # the NVIDIA pin table and dll-pins.txt stay NVIDIA-only

    def test_fsr_collect_is_cached_verified_and_opt_in(self):
        cache = self.root / 'cache'
        payload = b'amd denoiser stand-in'
        digest = hashlib.sha256(payload).hexdigest()
        (cache / 'amd').mkdir(parents=True)
        (cache / 'amd' / 'amd_fidelityfx_denoiser_dx12.dll').write_bytes(payload)
        fake = ('amd_fidelityfx_denoiser_dx12.dll', 'amd_fidelityfx_denoiser_dx12.dll', 'https://unused.invalid', digest)
        with patch.object(installer, 'FSR_DENOISER', fake), patch.object(installer, 'collect', wraps=installer.collect):
            with patch.object(installer, 'PINS', {}):
                self.assertEqual(installer.collect(cache, True, False), [])
                assets = installer.collect(cache, True, True)
        self.assertEqual([(a[1], a[2]) for a in assets], [('amd_fidelityfx_denoiser_dx12.dll', digest)])
        (cache / 'amd' / 'amd_fidelityfx_denoiser_dx12.dll').write_bytes(b'tampered')
        with patch.object(installer, 'FSR_DENOISER', fake), patch.object(installer, 'PINS', {}):
            with self.assertRaisesRegex(RuntimeError, 'Invalid offline cache'):
                installer.collect(cache, True, True)

    def test_ini_selection_of_fsr(self):
        ini = self.runtime / 'd4r.ini'
        cases = (
            (['[RayReconstruction]', 'Backend = fsr'], True),
            (['[RayReconstruction]', 'Backend = FSR ; note'], True),
            (['[RayReconstruction]', 'Backend = native'], False),
            (['[Other]', 'Backend = fsr'], False),
            ([';[RayReconstruction]', '; Backend = fsr'], False),
            ([''], False),
        )
        for lines, expected in cases:
            ini.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            self.assertEqual(installer.ini_selects_fsr(self.runtime), expected, lines)

    def test_select_fsr_backend_preserves_ini_and_backs_up(self):
        original = ('; keep\n[RayReconstruction]\n; note\nBackend = off ; trailing\nModelDir = auto\n[Debug]\nLog = normal\n').encode()
        ini = self.runtime / 'd4r.ini'
        ini.write_bytes(b'\xef\xbb\xbf' + original)
        backup = installer.enable_fsr(self.runtime, self.backups)
        self.assertEqual(backup.read_bytes(), b'\xef\xbb\xbf' + original)
        updated = ini.read_bytes()
        self.assertTrue(updated.startswith(b'\xef\xbb\xbf'))
        self.assertIn(b'Backend = fsr ; trailing', updated)
        self.assertEqual(updated.replace(b'Backend = fsr', b'Backend = off'), b'\xef\xbb\xbf' + original)
        self.assertIsNone(installer.enable_fsr(self.runtime, self.backups))
        self.assertEqual(len(list(self.backups.iterdir())), 1)
        ini.write_bytes(b'[Debug]\nLog = normal\n')
        installer.enable_fsr(self.runtime, self.backups)
        self.assertTrue(installer.ini_selects_fsr(self.runtime))
        ini.write_bytes(b'[RayReconstruction]')
        installer.enable_fsr(self.runtime, self.backups)
        self.assertTrue(installer.ini_selects_fsr(self.runtime))

    def test_offline_bad_cache_preserved(self):
        cache = self.root / 'cache.dll'
        cache.write_bytes(b'old')
        with self.assertRaisesRegex(RuntimeError, 'Invalid offline cache'):
            installer.verified_download(cache, 'https://unused.invalid', '0' * 64, True)
        self.assertEqual(cache.read_bytes(), b'old')

    def test_cache_backups_outside_game(self):
        with self.assertRaisesRegex(RuntimeError, 'outside game'):
            installer.external(self.runtime / 'cache', self.runtime.parent)
        installer.external(self.backups, self.runtime.parent)

    def test_tools_ignore_unlisted_private_products_and_reject_tamper(self):
        bundle = self.root / 'bundle'
        names = ('python/python.exe', 'recipe/tools/windows/build_rr_model.py',
                 'recipe/tools/windows/deploy_rr_model.py', 'recipe/engine/rr/native_kernels.txt')
        for name in names:
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(name.encode())
        private = bundle / 'private.weights'
        private.write_bytes(b'never copy')
        manifest = dict(kind='local-rr-model-tools', modelsBundled=False, nvidiaBinariesBundled=False,
                        files={name: installer.sha(bundle / name) for name in names})
        (bundle / 'rr-tools.json').write_text(json.dumps(manifest))
        target = installer.prepare_model_tools(bundle, self.root / 'cache')
        self.assertFalse((target / private.name).exists())
        (target / names[0]).write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'cache changed'):
            installer.prepare_model_tools(bundle, self.root / 'cache')


if __name__ == '__main__':
    unittest.main()
