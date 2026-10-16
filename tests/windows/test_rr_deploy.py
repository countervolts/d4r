import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/windows'))
from deploy_rr_model import select_native


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
