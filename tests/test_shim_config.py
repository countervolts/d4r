"""Marker polling defaults and the shim's model-override diagnostics."""
import configparser
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / 'tools/d4r_nvngx_shim.cpp'


class ShimConfigTests(unittest.TestCase):
    def test_marker_poll_defaults_agree(self):
        for name in ('packaging/d4r.ini', 'config/d4r.ini.default'):
            ini = configparser.ConfigParser()
            ini.read(ROOT / name)
            self.assertEqual(ini.getint('Interop', 'MarkerPollUs'), 200, name)
        source = SHIM.read_text()
        portable = re.search(r'portable_set\("D4R_SHIM_MARKER_POLL_US", poll.empty\(\) \? "(\d+)" : poll\);', source)
        runtime = re.search(r'env_uint\("D4R_SHIM_MARKER_POLL_US", (\d+)\)', source)
        self.assertIsNotNone(portable, 'portable marker polling fallback missing')
        self.assertIsNotNone(runtime, 'runtime marker polling fallback missing')
        self.assertEqual(int(portable[1]), 200)
        self.assertEqual(int(runtime[1]), 200)

    def test_engine_is_opt_in_and_configured_by_ini(self):
        for name in ('packaging/d4r.ini', 'config/d4r.ini.default'):
            ini = configparser.ConfigParser(inline_comment_prefixes=(';',))
            ini.read(ROOT / name)
            self.assertFalse(ini.getboolean('Engine', 'Enabled'), name)
            self.assertEqual(ini.get('Engine', 'ModelDir'), 'auto', name)
        source = SHIM.read_text()
        self.assertIn('portable_set("D4R_ENGINE", ini_flag(ini, "engine", "Enabled", 0) ? "1" : "0");', source)
        self.assertIn('env_uint("D4R_ENGINE", 0)', source)
        with tempfile.TemporaryDirectory() as temporary:
            ini = Path(temporary) / 'd4r.ini'
            env = {key: value for key, value in os.environ.items() if not key.startswith('D4R_')}
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            for text, enabled, model in (('', '0', None), ('[Engine]\nEnabled = true\n', '1', None),
                                         ('[Engine]\nEnabled = true\nModelDir = /models/k\n', '1', '/models/k')):
                ini.write_text(text)
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
                self.assertIn(f'export D4R_ENGINE={enabled}\n', result.stdout)
                self.assertEqual('export D4R_ENGINE_MODEL_DIR=/models/k\n' in result.stdout, model is not None)
            ini.write_text('[Engine]\nEnabled = maybe\n')
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)

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

    def test_production_model_override_logs_and_presets(self):
        # Compile the actual feature-creation preset block with fake NGX inputs
        # and a captured logger, so the test exercises the production message
        # and override behavior without requiring Windows, NGX or a GPU.
        source = SHIM.read_text()
        creation = source.index('static NgxResult create_feature(')
        start = source.index('    const char* presetNames[] =', creation)
        end = source.index('    feature->preset = presets[1];', start)
        block = source[start:end]
        runner_source = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <string>
static unsigned nextPreset = 21;
unsigned get_uint_or(void*, const char*, unsigned) { return nextPreset++; }
std::string env_string(const char* name) { const char* value = getenv(name); return value ? value : ""; }
void logf(const char* format, ...) {
    va_list args; va_start(args, format); vprintf(format, args); va_end(args); puts("");
}
int main() {
    void* parameters = nullptr;
''' + block + r'''
    printf("presets:"); for (unsigned preset : presets) printf(" %u", preset); puts("");
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / 'model-override'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-x', 'c++', '-', '-o', str(runner)],
                           input=runner_source, text=True, capture_output=True, check=True)
            env = {key: value for key, value in os.environ.items() if key != 'D4R_DLSS_PRESET'}
            for preset, model in ((5, 'E'), (11, 'K'), (12, 'L'), (13, 'M')):
                with self.subTest(model=model):
                    output = subprocess.check_output([str(runner)], env=dict(env, D4R_DLSS_PRESET=str(preset)), text=True)
                    self.assertIn(f'D4R_DLSS_PRESET={preset} (model {model}) overrides game/OptiScaler presets', output)
                    self.assertIn('[DLAA=21 Quality=22 Balanced=23 Performance=24 UltraPerformance=25 UltraQuality=26]', output)
                    self.assertIn(f'using preset={preset} for all quality modes', output)
                    self.assertIn('presets:' + f' {preset}' * 6, output)
            for value in (None, ''):
                current = env if value is None else dict(env, D4R_DLSS_PRESET=value)
                output = subprocess.check_output([str(runner)], env=current, text=True)
                self.assertNotIn('model override', output)
                self.assertEqual(output.strip(), 'presets: 21 22 23 24 25 26')


if __name__ == '__main__':
    unittest.main()
