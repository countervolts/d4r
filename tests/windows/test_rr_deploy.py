import sys
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/windows'))
from deploy_rr_model import select_native, model_arithmetic


class ModelTargetTests(unittest.TestCase):
    def test_rdna4_targets_require_exact_match(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            for target in ('gfx1200', 'gfx1201'):
                identity = {'preset': 'D', 'architecture': target, 'arithmetic': 'exact-gfx12'}
                (runtime / 'gpu-target.json').write_text(json.dumps({'architecture': target}))
                self.assertEqual(model_arithmetic(identity, runtime), 'gfx12')
                identity['architecture'] = 'gfx1201' if target == 'gfx1200' else 'gfx1200'
                with self.assertRaisesRegex(RuntimeError, 'does not match'):
                    model_arithmetic(identity, runtime)

    def test_rdna4_requires_target_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = {'preset': 'E', 'architecture': 'gfx1201', 'arithmetic': 'exact-gfx12'}
            with self.assertRaisesRegex(RuntimeError, 'gpu-target.json'):
                model_arithmetic(identity, Path(directory))

    def test_wrong_arithmetic_rejected_and_legacy_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = {'preset': 'E', 'architecture': 'gfx1101', 'arithmetic': 'exact-gfx11'}
            self.assertEqual(model_arithmetic(identity, Path(directory)), 'gfx11')
            identity['architecture'] = 'gfx1201'
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                model_arithmetic(identity, Path(directory))


class ConfigSelectionTests(unittest.TestCase):
    def test_comments_bom_and_crlf_survive(self):
        source = b'\xef\xbb\xbf[DLSS]\r\nModel = K ; keep SR\r\n[RayReconstruction]\r\nBackend = off ; keep comment\r\nModelDir = auto # keep path note\r\n'
        result = select_native(source, 'engine/rr-d-test')
        self.assertEqual(result, source.replace(b'Backend = off', b'Backend = native').replace(b'ModelDir = auto', b'ModelDir = engine/rr-d-test'))
        self.assertEqual(select_native(result, 'engine/rr-d-test'), result)

    def test_missing_section_preserves_other_settings(self):
        source = b'[DLSS]\nModel=M\n; custom note'
        result = select_native(source, 'engine/rr-e-test')
        self.assertTrue(result.startswith(source))
        self.assertIn(b'[RayReconstruction]\nBackend = native\nModelDir = engine/rr-e-test\n', result)
        self.assertEqual(select_native(result, 'engine/rr-e-test'), result)

    def test_duplicate_sections_cannot_restore_old_selection(self):
        source = b'[RayReconstruction]\nBackend=off\n[Other]\nKeep=yes\n[rayreconstruction]\nModelDir=old-model\n'
        result = select_native(source, 'engine/rr-d-test')
        self.assertNotIn(b'old-model', result)
        self.assertNotIn(b'Backend=off', result)
        self.assertEqual(result.count(b'engine/rr-d-test'), 2)
        self.assertIn(b'[Other]\nKeep=yes\n', result)
        self.assertEqual(select_native(result, 'engine/rr-d-test'), result)


if __name__ == '__main__':
    unittest.main()
