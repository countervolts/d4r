"""Engine defaults, preset-selection behavior and polling configuration precedence."""
import configparser
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ShimConfigTests(unittest.TestCase):
    def test_native_motion_vectors_require_opt_in(self):
        for name in ('packaging/d4r.ini', 'config/d4r.ini.default'):
            config = configparser.ConfigParser(inline_comment_prefixes=(';',))
            config.read(ROOT / name)
            self.assertFalse(config.getboolean('Interop', 'NativeMotionVectors'))
        with tempfile.TemporaryDirectory() as directory:
            ini = Path(directory) / 'd4r.ini'
            env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
            command = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            for value, expected in ((None, '0'), ('false', '0'), ('true', '1'), ('auto', '0')):
                ini.write_text('[Interop]\n' +
                               (f'NativeMotionVectors = {value}\n' if value is not None else ''))
                result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                self.assertIn(f'export D4R_NATIVE_MV_LIVE={expected}\n', result.stdout)
            ini.write_text('[Interop]\nNativeMotionVectors = invalid\n')
            self.assertEqual(subprocess.run(command, env=env, capture_output=True).returncode, 2)

    def test_portable_native_motion_opt_in_blocks_old_environment_override(self):
        source = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        begin = source.index('    const bool live=', source.index('static void record_native_gpu_check('))
        policy = source[begin:source.index(';', begin) + 1]
        program = r'''
#include <cassert>
struct { bool active, nativeMotionVectors; } g_portable;
struct { bool rayReconstruction; unsigned flags; } feature;
static unsigned requested;
static unsigned env_uint(const char*, unsigned) { return requested; }
static bool enabled() {
''' + policy + r'''
    return live;
}
int main() {
    feature = {false, 2};
    requested = 1;
    g_portable = {true, false}; assert(!enabled());
    g_portable = {true, true}; assert(enabled());
    feature.rayReconstruction = true; assert(!enabled());
    feature = {false, 0}; assert(!enabled());
    feature.flags = 2;
    requested = 0; assert(!enabled());
    requested = 1;
    g_portable = {false, false}; assert(enabled());
}
'''
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / 'native-motion-policy'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-x', 'c++',
                            '-', '-o', str(runner)], input=program, text=True, capture_output=True, check=True)
            subprocess.run([str(runner)], check=True)

    def test_vkhip_is_on_by_default_and_environment_takes_precedence(self):
        for name in ('packaging/d4r.ini', 'config/d4r.ini.default'):
            config = configparser.ConfigParser(inline_comment_prefixes=(';',))
            config.read(ROOT / name)
            self.assertTrue(config.getboolean('Engine', 'ExperimentalVkHip'))
        with tempfile.TemporaryDirectory() as directory:
            ini=Path(directory)/'d4r.ini'
            env={k:v for k,v in os.environ.items() if not k.startswith('D4R_')}
            command=['python3',str(ROOT/'scripts/d4r_config.py'),'--config',str(ini)]
            for text,want in (('[Engine]\nEnabled=true\nExperimentalVkHip=false\n','0'),
                              ('[Engine]\nEnabled=true\nExperimentalVkHip=true\n','1'),('','1'),
                              ('[Engine]\nEnabled=auto\n','1')):
                ini.write_text(text)
                result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
                self.assertIn('export D4R_ENGINE_VKHIP='+want+'\n',result.stdout)
                # vkhip does not force K onto the engine: auto still leaves the RDNA3 fallback to the shim
                self.assertNotIn('export D4R_ENGINE_K=1\n',result.stdout)
            result=subprocess.run(command,env=dict(env,D4R_ENGINE_VKHIP='0'),text=True,capture_output=True,check=True)
            self.assertNotIn('export D4R_ENGINE_VKHIP=',result.stdout)
            ini.write_text('[Engine]\nExperimentalVkHip=invalid\n')
            self.assertEqual(subprocess.run(command,env=env,capture_output=True).returncode,2)

    def test_profile_log_interval_validation_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            ini = Path(temporary) / 'd4r.ini'
            env = {key: value for key, value in os.environ.items() if not key.startswith('D4R_')}
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            for value, expected in ((None, None), ('auto', None), ('1', '1'), ('60', '60'),
                                    ('060', '60'), ('120', '120'), ('4294967295', '4294967295')):
                ini.write_text('[Debug]\nProfile = true\n' +
                               (f'ProfileLogEvery = {value}\n' if value is not None else ''))
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
                self.assertIn('export D4R_PROFILE=1\n', result.stdout)
                if expected is None:
                    self.assertNotIn('export D4R_PROFILE_LOG_EVERY=', result.stdout)
                else:
                    self.assertIn(f'export D4R_PROFILE_LOG_EVERY={expected}\n', result.stdout)
            ini.write_text('[Debug]\nProfileLogEvery = 60\n')
            result = subprocess.run(cmd, env=dict(env, D4R_PROFILE_LOG_EVERY='120'),
                                    text=True, capture_output=True, check=True)
            self.assertNotIn('export D4R_PROFILE_LOG_EVERY=', result.stdout)
            for value in ('0', '-1', '0.5', 'invalid', '4294967296'):
                ini.write_text(f'[Debug]\nProfileLogEvery = {value}\n')
                result = subprocess.run(cmd, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 2, value)
        for name in ('packaging/d4r.ini', 'config/d4r.ini.default'):
            ini = configparser.ConfigParser(inline_comment_prefixes=(';',))
            ini.read(ROOT / name)
            self.assertEqual(ini.getint('Debug', 'ProfileLogEvery'), 60, name)

    def test_engine_defaults_and_ini_overrides(self):
        for name, expected in (('packaging/d4r.ini', 'auto'), ('config/d4r.ini.default', 'false')):
            ini = configparser.ConfigParser(inline_comment_prefixes=(';',))
            ini.read(ROOT / name)
            self.assertEqual(ini.get('Engine', 'Enabled'), expected, name)
            self.assertEqual(ini.get('Engine', 'ModelDir'), 'auto', name)
        with tempfile.TemporaryDirectory() as temporary:
            ini = Path(temporary) / 'd4r.ini'
            env = {key: value for key, value in os.environ.items() if not key.startswith('D4R_')}
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            for text, enabled, model in (('', '0', None), ('[Engine]\nEnabled = auto\n', '0', None),
                                         ('[Engine]\nEnabled = false\n', '0', None),
                                         ('[Engine]\nEnabled = true\n', '1', None),
                                         ('[Engine]\nEnabled = true\nModelDir = /models/k\n', '1', '/models/k')):
                ini.write_text(text)
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
                self.assertIn(f'export D4R_ENGINE={enabled}\n', result.stdout)
                self.assertEqual('export D4R_ENGINE_MODEL_DIR=/models/k\n' in result.stdout, model is not None)
            ini.write_text('[Engine]\nEnabled = maybe\n')
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_auto_engine_detects_models_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            ini = root / 'd4r.ini'
            ini.write_text('[Engine]\nEnabled = auto\n')
            (root / 'engine/rr-d-gfx11').mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            output = subprocess.check_output(cmd, env=env, text=True)
            self.assertIn('export D4R_ENGINE=1\n', output)
            self.assertIn('export D4R_ENGINE_K=0\n', output)
            output = subprocess.check_output(cmd, env=dict(env, D4R_ENGINE='0'), text=True)
            self.assertNotIn('export D4R_ENGINE=', output)

    def test_launcher_marker_poll_defaults_overrides_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            ini = Path(temporary) / 'd4r.ini'
            env = {key: value for key, value in os.environ.items() if not key.startswith('D4R_')}
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            for value, expected in ((None, None), ('auto', None), ('200', '200'), ('0', '0'), ('75', '75')):
                ini.write_text('[Interop]\n' + (f'MarkerPollUs = {value}\n' if value is not None else ''))
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
                if expected is None:
                    self.assertNotIn('export D4R_SHIM_MARKER_POLL_US=', result.stdout)
                else:
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


class OptiScalerPresetFollowerTests(unittest.TestCase):
    """OptiScaler's Render Presets Override wins over [DLSS] Model once it changes the game's presets."""

    def run_policy(self, body):
        source = r'''
#include "d4r_render_presets.h"
#include <cassert>
#include <cstring>
static bool same(const unsigned int (&a)[6], const unsigned int (&b)[6]) { return std::memcmp(a, b, sizeof(a)) == 0; }
int main() {
''' + body + '\n}\n'
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / 'follower'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                            '-I', str(ROOT / 'tools'), '-x', 'c++', '-', '-o', str(runner)],
                           input=source, text=True, capture_output=True, check=True)
            subprocess.run([str(runner)], check=True, timeout=5)

    def test_game_presets_keep_the_d4r_model_until_optiscaler_changes_them(self):
        self.run_policy(r'''
    D4rPresetFollower follower;
    const unsigned int game[6] = {0, 0, 0, 0, 0, 0};
    assert(!follower.follow(game, false));          // first creation: the game's presets
    assert(!follower.follow(game, false));          // a resolution or quality change recreates with the same
    const unsigned int m[6] = {13, 13, 13, 13, 13, 13};
    assert(follower.follow(m, false));              // OptiScaler: override M, Apply Changes
    const unsigned int e[6] = {5, 5, 5, 5, 5, 5};
    assert(follower.follow(e, false));              // then E
    assert(!follower.follow(game, false));          // override unticked: OptiScaler restores the game's presets
''')

    def test_game_presets_that_differ_between_callers_are_baselines(self):
        self.run_policy(r'''
    // SILENT HILL Townfall: the first feature (flags 0xb) has no presets, gameplay features (0x49) ask for F
    D4rPresetFollower follower;
    const unsigned int none[6] = {0, 0, 0, 0, 0, 0};
    const unsigned int f[6] = {6, 6, 6, 6, 6, 6};
    assert(!follower.follow(none, false, 0xb));
    assert(!follower.follow(f, false, 0x49));       // the game's own hint, not the player's choice
    assert(!follower.follow(f, false, 0x49));       // quality changes recreate with the same presets
    const unsigned int k[6] = {11, 11, 11, 11, 11, 11};
    assert(follower.follow(k, false, 0x49));        // OptiScaler: override K, Apply Changes
    assert(follower.follow(k, false, 0x4b));        // still applied when the game recreates with new flags
    assert(!follower.follow(f, false, 0x49));       // override unticked: back to the game's presets
    const unsigned int h[6] = {6, 6, 6, 6, 6, 6};
    assert(!follower.follow(h, false, 0x41));       // new flags after unticking: the game's presets again
''')

    def test_explicit_game_presets_are_the_baseline_too(self):
        self.run_policy(r'''
    D4rPresetFollower follower;
    const unsigned int game[6] = {11, 11, 11, 13, 12, 11};
    assert(!follower.follow(game, false));
    const unsigned int perMode[6] = {11, 11, 11, 13, 13, 11};  // OptiScaler's per-mode presets
    assert(follower.follow(perMode, false));
''')

    def test_ini_override_is_followed_from_the_first_creation(self):
        self.run_policy(r'''
    D4rPresetFollower follower;
    const unsigned int k[6] = {11, 11, 11, 11, 11, 11};
    assert(follower.follow(k, true));
    assert(follower.follow(k, true));
''')

    def test_default_and_latest_slots_take_the_d4r_model(self):
        self.run_policy(r'''
    unsigned int presets[6] = {0, 13, kD4rPresetLatest, 5, 12, 0};
    d4r_apply_followed_presets(presets, d4r_parse_render_preset_override("11"));
    const unsigned int expected[6] = {11, 13, 11, 5, 12, 11};
    assert(same(presets, expected));
    unsigned int untouched[6] = {0, 13, 0, 5, 12, 0};
    const unsigned int copy[6] = {0, 13, 0, 5, 12, 0};
    d4r_apply_followed_presets(untouched, d4r_parse_render_preset_override(""));
    assert(same(untouched, copy));
''')

    def test_flagged_and_retired_presets_read_as_default(self):
        self.run_policy(r'''
    // Alan Wake 2 (Streamline): E and F with the high bit set, then OptiScaler recreates with zeros
    unsigned int game[6] = {0x80000006u, 0x80000005u, 0x80000005u, 0x80000005u, 0x80000006u, 0x80000006u};
    d4r_sanitize_presets(game, false);
    const unsigned int zeros[6] = {0, 0, 0, 0, 0, 0};
    assert(same(game, zeros));
    D4rPresetFollower follower;
    assert(!follower.follow(game, false));
    unsigned int reset[6] = {0, 0, 0, 0, 0, 0x80000006u};
    d4r_sanitize_presets(reset, false);
    assert(!follower.follow(reset, false));         // not a choice made in OptiScaler
    unsigned int mixed[6] = {1, 4, 7, 11, 13, kD4rPresetLatest};
    d4r_sanitize_presets(mixed, false);
    const unsigned int expected[6] = {0, 0, 0, 11, 13, kD4rPresetLatest};
    assert(same(mixed, expected));
    unsigned int rr[6] = {4, 5, 6, 11, 0x80000004u, 0};
    d4r_sanitize_presets(rr, true);
    const unsigned int rrExpected[6] = {4, 5, 6, 0, 0, 0};
    assert(same(rr, rrExpected));
''')

    def test_quality_mode_selects_ngx_preset_slot(self):
        self.run_policy(r'''
    // slots: DLAA, Quality, Balanced, Performance, UltraPerformance, UltraQuality
    assert(d4r_preset_slot(0) == 3);   // MaxPerf
    assert(d4r_preset_slot(1) == 2);   // Balanced
    assert(d4r_preset_slot(2) == 1);   // MaxQuality
    assert(d4r_preset_slot(3) == 4);   // UltraPerformance
    assert(d4r_preset_slot(4) == 5);   // UltraQuality
    assert(d4r_preset_slot(5) == 0);   // DLAA
    assert(d4r_preset_slot(-1) == 1 && d4r_preset_slot(9) == 1);
''')



if __name__ == '__main__':
    unittest.main()
