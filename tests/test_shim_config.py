"""Marker polling configuration validation and environment precedence."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ShimConfigTests(unittest.TestCase):
    def test_launcher_marker_poll_defaults_overrides_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            ini = Path(temporary) / 'd4r.ini'
            env = {key: value for key, value in os.environ.items() if not key.startswith('D4R_')}
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            for value, expected in ((None, '200'), ('auto', '200'), ('200', '200'), ('0', '0'), ('75', '75')):
                ini.write_text('[Interop]\n' + (f'MarkerPollUs = {value}\n' if value is not None else ''))
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
                self.assertIn(f'export D4R_SHIM_MARKER_POLL_US={expected}\n', result.stdout)
            # An explicit environment variable keeps its existing precedence.
            result = subprocess.run(cmd, env=dict(env, D4R_SHIM_MARKER_POLL_US='35'),
                                    text=True, capture_output=True, check=True)
            self.assertNotIn('export D4R_SHIM_MARKER_POLL_US=', result.stdout)
            for value in ('-1', '0.5', 'invalid'):
                ini.write_text(f'[Interop]\nMarkerPollUs = {value}\n')
                result = subprocess.run(cmd, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('MarkerPollUs must be a number of microseconds', result.stderr)

if __name__ == '__main__':
    unittest.main()
