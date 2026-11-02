"""Per-game opt-in and rollback boundaries, plus the real NVAPI router under Wine."""
import configparser
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('features', ROOT / 'packaging/d4r-features.py')
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


class FeatureConfigTests(unittest.TestCase):
    def fixture(self, directory):
        game = Path(directory)
        runtime = game / 'plugins/d4r'
        runtime.mkdir(parents=True)
        ini = game / 'OptiScaler.ini'
        ini.write_bytes(b'\xef\xbb\xbf; keep\r\n[Libraries]\r\nNvngxPath=plugins\\d4r.asi\r\n'
                        b'[DLSS]\r\nEnabled=true\r\n[FrameGen]\r\nEnabled=false\r\n'
                        b'FGInput=nofg\r\nFGOutput=nofg\r\n[Menu]\r\nShowFps=true\r\n')
        for name in ('d4r_nvapi64.dll', 'fakenvapi.dll', 'libxell.dll', 'libxess_fg.dll',
                     'amd_fidelityfx_framegeneration_dx12.dll'):
            (runtime / name).write_bytes(b'fixture')
        return ini, runtime

    def test_xefg_and_disable_preserve_ngx_and_backup_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            ini, runtime = self.fixture(directory)
            original = ini.read_bytes()
            backup = features.configure(ini, runtime, 'xefg', 'dlssg', 'auto')
            self.assertEqual(backup.read_bytes(), original)
            data = ini.read_bytes()
            self.assertEqual(data.count(b'\r\n'), data.count(b'\n'))
            self.assertTrue(data.startswith(b'\xef\xbb\xbf; keep\r\n'))
            cfg = configparser.ConfigParser()
            cfg.read_string(data.decode('utf-8-sig'))
            self.assertEqual(cfg['Libraries']['NvngxPath'], r'plugins\d4r.asi')
            self.assertEqual(cfg['Libraries']['NvapiPath'], features.winpath(runtime / 'd4r_nvapi64.dll'))
            self.assertEqual(cfg['Libraries']['OptiDllPath'], features.winpath(runtime))
            self.assertEqual(cfg['Libraries']['XeFGPath'], features.winpath(runtime))
            self.assertEqual(cfg['DLSS']['Enabled'], 'true')
            self.assertEqual(cfg['Menu']['ShowFps'], 'true')
            self.assertEqual(cfg['FrameGen']['FGInput'], 'dlssg')
            self.assertIsNone(features.configure(ini, runtime, 'xefg', 'dlssg', 'auto'))
            features.configure(ini, runtime, 'off', None, 'proton')
            cfg.read_string(ini.read_bytes().decode('utf-8-sig'))
            self.assertEqual(cfg['FrameGen']['FGOutput'], 'nofg')
            self.assertEqual(cfg['NvApi']['OverrideNvapiDll'], 'false')
            self.assertEqual(cfg['Libraries']['NvngxPath'], r'plugins\d4r.asi')

    def test_rejected_requests_do_not_touch_config_or_create_backups(self):
        for case in ('missing', 'duplicate', 'noinput', 'incompatible', 'route'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                ini, runtime = self.fixture(directory)
                if case == 'missing':
                    (runtime / 'libxess_fg.dll').unlink()
                if case == 'duplicate':
                    ini.write_bytes(ini.read_bytes() + b'[libraries]\r\nNvngxPath=other\r\n')
                if case == 'route':
                    ini.write_bytes(ini.read_bytes().replace(b'plugins\\d4r.asi', b'auto'))
                original = ini.read_bytes()
                with self.assertRaises((ValueError, configparser.Error)):
                    features.configure(ini, runtime, 'xefg', None if case == 'noinput' else 'dlssg',
                                       'proton' if case == 'incompatible' else 'auto')
                self.assertEqual(ini.read_bytes(), original)
                self.assertFalse(list(ini.parent.glob('*.d4r-features-backup-*')))

    def test_fsrfg_and_standalone_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            ini, runtime = self.fixture(directory)
            features.configure(ini, runtime, 'fsrfg', 'upscaler', 'auto')
            cfg = configparser.ConfigParser()
            cfg.read_string(ini.read_bytes().decode('utf-8-sig'))
            self.assertEqual(cfg['Libraries']['FfxDx12FGPath'],
                             features.winpath(runtime / 'amd_fidelityfx_framegeneration_dx12.dll'))
            features.configure(ini, runtime, 'off', None, 'auto')
            cfg.read_string(ini.read_bytes().decode('utf-8-sig'))
            self.assertEqual(cfg['NvApi']['OverrideNvapiDll'], 'true')
            self.assertEqual(cfg['FrameGen']['Enabled'], 'false')

    def test_v10_uses_bundled_providers_without_external_nvapi_router(self):
        for fg in ('xefg', 'fsrfg', 'off'):
            with self.subTest(fg=fg), tempfile.TemporaryDirectory() as directory:
                ini, runtime = self.fixture(directory)
                original = ini.read_bytes() + b'[fakenvapi]\r\nUseFakenvapi=false\r\n'
                ini.write_bytes(original)
                providers = ini.parent / 'OptiScaler'
                providers.mkdir()
                for name in ('libxell.dll', 'libxess_fg.dll', 'amd_fidelityfx_loader_dx12.dll',
                             'amd_fidelityfx_framegeneration_dx12.dll'):
                    (providers / name).write_bytes(b'fixture')
                shutil.rmtree(runtime)
                backup = features.configure(ini, runtime, fg, 'dlssg' if fg != 'off' else None, 'auto')
                self.assertEqual(backup.read_bytes(), original)
                cfg = configparser.ConfigParser()
                cfg.read_string(ini.read_bytes().decode('utf-8-sig'))
                self.assertEqual(cfg['fakenvapi']['UseFakenvapi'], 'false')
                self.assertEqual(cfg['fakenvapi']['LowLatencyOutput'], 'auto')
                self.assertFalse(cfg.has_option('Libraries', 'NvapiPath'))
                self.assertFalse(cfg.has_section('NvApi'))
                self.assertEqual(cfg['Libraries']['NvngxPath'], r'plugins\d4r.asi')
                self.assertIsNone(features.configure(ini, runtime, fg, 'dlssg' if fg != 'off' else None, 'auto'))
                features.configure(ini, runtime, 'off', None, 'proton')
                cfg.read_string(ini.read_bytes().decode('utf-8-sig'))
                self.assertEqual(cfg['fakenvapi']['LowLatencyOutput'], '6')

    def test_v10_missing_fg_provider_leaves_config_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            ini, runtime = self.fixture(directory)
            ini.write_bytes(ini.read_bytes() + b'[fakenvapi]\r\nUseFakenvapi=false\r\n')
            original = ini.read_bytes()
            with self.assertRaisesRegex(ValueError, 'extract the complete OptiScaler/'):
                features.configure(ini, runtime, 'xefg', 'dlssg', 'auto')
            self.assertEqual(ini.read_bytes(), original)
            self.assertFalse(list(ini.parent.glob('*.d4r-features-backup-*')))


@unittest.skipUnless(shutil.which('wine') and shutil.which('x86_64-w64-mingw32-g++'), 'needs Wine and MinGW')
class NvapiRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='d4r-nvapi-test-')
        cls.directory = Path(cls.temporary.name)
        cls.prefix = cls.directory / 'prefix'
        cls.env = dict(os.environ, WINEPREFIX=str(cls.prefix), WINEDEBUG='-all',
                       WINEDLLOVERRIDES='nvapi64=;winemenubuilder.exe=d')
        cls.compiler = 'x86_64-w64-mingw32-g++'
        def compile_file(source, output, *flags):
            subprocess.run([cls.compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-static',
                            *flags, str(source), '-o', str(output)], capture_output=True, text=True, check=True)
        cls.router = cls.directory / 'd4r_nvapi64.dll'
        compile_file(ROOT / 'tools/d4r_nvapi_router.cpp', cls.router, '-shared')
        stub = cls.directory / 'provider.cpp'
        stub.write_text(r'''
#include <windows.h>
static int value() { return PROVIDER; }
static int init() { return PROVIDER == 22 ? -5 : 0; } // optional provider must not override real status
extern "C" __declspec(dllexport) void* nvapi_QueryInterface(unsigned id) {
    if (id == 0x0150e828 || id == 0xd22bdd7e) return reinterpret_cast<void*>(&init);
    if (id == 0xffffffff) return nullptr;
    if (PROVIDER == 22 && id == 0x1a587f9c) return nullptr; // older latency provider
    return reinterpret_cast<void*>(&value);
}
''')
        cls.real = cls.directory / 'real.dll'
        cls.fake = cls.directory / 'fakenvapi.dll'
        compile_file(stub, cls.real, '-shared', '-DPROVIDER=11')
        compile_file(stub, cls.fake, '-shared', '-DPROVIDER=22')
        probe = cls.directory / 'probe.cpp'
        probe.write_text(r'''
#include <windows.h>
#include <cstdlib>
int wmain(int argc, wchar_t** argv) {
    if (argc != 3) return 1;
    auto dll = LoadLibraryW(argv[1]);
    if (!dll) return 2;
    using Query = void*(*)(unsigned);
    auto query = reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(dll, "nvapi_QueryInterface")));
    if (!query) return 3;
    using Fn = int(*)();
    auto call = [&](unsigned id) { auto fn = reinterpret_cast<Fn>(query(id)); return fn ? fn() : -999; };
    int mode = _wtoi(argv[2]);
    if (mode == 2) return query(0xdeadbeef) || query(0x0150e828) ? 4 : 0;
    if (call(0x0150e828) != 0) return 5;
    // GPU enumeration, architecture and unknown extensions must keep real handles.
    for (unsigned id : {0xe5ac921fU, 0xd8265d24U, 0xdeadbeefU, 0x01053fa5U})
        if (call(id) != 11) return 6;
    for (unsigned id : {0xac1ca9e0U, 0x852cd1d2U, 0xd9984c05U, 0x13c98f73U, 0x21412141U})
        if (call(id) != (mode == 0 ? 22 : 11)) return 7;
    if (call(0x1a587f9c) != 11) return 10; // missing individual export falls back too
    if (query(0xffffffff)) return 8;
    return call(0xd22bdd7e) == 0 ? 0 : 9;
}
'''.replace('#include <cstdlib>', '#include <cstdlib>\n#include <initializer_list>'))
        cls.probe = cls.directory / 'probe.exe'
        compile_file(probe, cls.probe, '-municode')
        # Initialize only a disposable prefix. No game, user prefix or system NVAPI changes.
        result = subprocess.run(['wine', str(cls.probe)], env=cls.env, capture_output=True, timeout=60)
        if not (cls.prefix / 'drive_c/windows/system32').is_dir():
            raise RuntimeError(result.stderr.decode(errors='replace'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_topology_stays_real_and_missing_latency_falls_back(self):
        for mode in (0, 1, 2):
            with self.subTest(mode=mode):
                target = self.directory / f'mode{mode}'
                target.mkdir()
                shutil.copy2(self.router, target / self.router.name)
                real = self.prefix / 'drive_c/windows/system32/d4r_nvapi64_real.dll'
                if mode == 2:
                    real.unlink()
                else:
                    shutil.copy2(self.real, real)
                if mode == 0:
                    shutil.copy2(self.fake, target / 'fakenvapi.dll')
                router_path = 'Z:' + str(target / self.router.name).replace('/', '\\')
                result = subprocess.run(['wine', str(self.probe), router_path, str(mode)],
                                        env=self.env, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
