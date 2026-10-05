"""Preset-selection behavior and polling configuration precedence."""
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


class RenderPresetPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.runner = Path(cls.directory.name) / 'render-presets'
        source = r'''
#include "d4r_render_presets.h"
#include <cstdio>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool rr = argv[1][0] == '1';
    unsigned int presets[6] = {21, 22, 23, 24, 25, 26};
    if (rr) {
        for (unsigned int& preset : presets) preset += 10;
    }
    const auto overrideValue = d4r_parse_render_preset_override(
        std::getenv(d4r_render_preset_variable(rr)));
    overrideValue.apply(presets);
    for (unsigned int preset : presets) std::printf("%u ", preset);
    std::puts("");
}
'''
        subprocess.run(
            ['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
             '-I', str(ROOT / 'tools'), '-x', 'c++', '-', '-o', str(cls.runner)],
            input=source, text=True, capture_output=True, check=True)

    def select(self, rr=False, **overrides):
        env = dict(os.environ)
        for name in ('D4R_DLSS_PRESET', 'D4R_RR_PRESET'):
            env.pop(name, None)
        env.update(overrides)
        result = subprocess.run(
            [str(self.runner), '1' if rr else '0'], env=env,
            text=True, capture_output=True, check=True, timeout=5)
        return [int(value) for value in result.stdout.split()]

    def test_sr_models_override_every_quality_mode(self):
        for value in (5, 11, 12, 13):
            with self.subTest(preset=value):
                self.assertEqual(
                    self.select(D4R_DLSS_PRESET=str(value), D4R_RR_PRESET='4'),
                    [value] * 6)

    def test_rr_models_do_not_use_the_sr_override(self):
        for value in (4, 5):
            with self.subTest(preset=value):
                self.assertEqual(
                    self.select(rr=True, D4R_RR_PRESET=str(value), D4R_DLSS_PRESET='13'),
                    [value] * 6)

    def test_unset_override_preserves_quality_specific_game_presets(self):
        for rr, foreign in ((False, 'D4R_RR_PRESET'), (True, 'D4R_DLSS_PRESET')):
            with self.subTest(rr=rr):
                expected = list(range(31, 37) if rr else range(21, 27))
                self.assertEqual(self.select(rr=rr), expected)
                self.assertEqual(self.select(rr=rr, **{foreign: '13'}), expected)

    def test_empty_override_does_not_fall_back_to_the_other_feature(self):
        self.assertEqual(
            self.select(D4R_DLSS_PRESET='', D4R_RR_PRESET='5'),
            list(range(21, 27)))
        self.assertEqual(
            self.select(rr=True, D4R_RR_PRESET='', D4R_DLSS_PRESET='13'),
            list(range(31, 37)))

    def test_explicit_default_preset_overrides_game_presets(self):
        self.assertEqual(self.select(D4R_DLSS_PRESET='0'), [0] * 6)
        self.assertEqual(self.select(rr=True, D4R_RR_PRESET='0'), [0] * 6)


if __name__ == '__main__':
    unittest.main()
