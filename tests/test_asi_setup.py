"""Native ASI configuration: backups, existing plugins, loader contract and automatic dispatch."""
import configparser
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('wine') and shutil.which('x86_64-w64-mingw32-g++'), 'Wine/MinGW required')
class AsiSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='d4r-asi-tests-')
        cls.directory = Path(cls.temporary.name)
        cls.helper = cls.directory / 'd4r-setup.exe'
        cls.probe = cls.directory / 'probe.exe'
        for source, output in (('d4r_asi_setup.cpp', cls.helper), ('d4r_asi_probe.cpp', cls.probe)):
            subprocess.run(['x86_64-w64-mingw32-g++', '-std=c++20', '-O2', '-Wall', '-Wextra',
                            '-Werror', '-static', '-municode', str(ROOT / 'tools' / source), '-o', str(output)],
                           check=True, capture_output=True)
        # Hold the second destination open without delete sharing. The helper
        # must restore the first DLL when the second replacement fails.
        lock_source = cls.directory / 'locked-dll.cpp'
        lock_source.write_text(r'''
#include <windows.h>
#include <string>
int wmain(int argc, wchar_t** argv) {
    if (argc != 5) return 2;
    HANDLE locked = CreateFileW(argv[2], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (locked == INVALID_HANDLE_VALUE) return 3;
    std::wstring command = L"\"" + std::wstring(argv[1]) + L"\" \"" + argv[3] + L"\" \"" + argv[4] + L"\"";
    STARTUPINFOW startup = {}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {};
    if (!CreateProcessW(argv[1], command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &process)) return 4;
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 0; GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread); CloseHandle(process.hProcess); CloseHandle(locked);
    return code == 1 ? 0 : 5;
}
''')
        cls.lock_probe = cls.directory / 'locked-dll.exe'
        subprocess.run(['x86_64-w64-mingw32-g++', '-std=c++20', '-O2', '-Wall', '-Wextra',
                        '-Werror', '-static', '-municode', str(lock_source), '-o', str(cls.lock_probe)],
                       check=True, capture_output=True)
        cls.env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
        cls.env.update(WINEPREFIX=str(cls.directory / 'prefix'), WINEDEBUG='-all', WINEARCH='win64',
                       WINEDLLOVERRIDES='winemenubuilder.exe=d', D4R_ASI_SETUP_QUIET='1')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def winpath(path):
        return 'Z:' + str(path).replace('/', '\\')

    def fixture(self, game, source):
        runtime = game / 'custom plugins/d4r'
        runtime.mkdir(parents=True)
        (game / 'Game.exe').touch()
        (game / 'winmm.dll').write_bytes(b'existing OptiScaler loader')
        (game / 'OptiScaler.ini').write_bytes(source)
        plugin = game / 'custom plugins/d4r.asi'
        plugin.write_bytes(b'ASI fixture')
        (plugin.parent / 'Other.asi').write_bytes(b'other mod')
        for name in ('nvcuda.dll', 'nvngx_dlss.dll', 'ngx/_nvngx.dll',
                     'zluda/libcuda.so', 'rocm/lib/libamdhip64.so.7'):
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (runtime / 'd4r.ini').write_text('[DLSS]\nModel=K\n')
        shutil.copy2(self.helper, runtime / 'd4r-setup.exe')
        for name in ('d3d12.dll', 'd4r12core.dll'):
            (runtime / name).write_bytes(('new ' + name).encode())
            (game / name).write_bytes(('old ' + name).encode())
        (game / 'D3D12Core.dll').write_bytes(b'original Microsoft SDK core')
        return plugin

    def setup(self, game, plugin):
        return subprocess.run(['wine', str(self.helper), self.winpath(game), self.winpath(plugin)],
                              env=self.env, cwd=self.directory, text=True, capture_output=True, timeout=60)

    def test_custom_plugin_folder_preservation_backup_and_retry(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp) / 'Game with spaces'
            original = (b'; keep this comment\r\n[Plugins]\r\nPath=custom plugins\r\n'
                        b'LoadAsiPlugins=true\r\nLoadReShade=true\r\n'
                        b'[FrameGen]\r\nFGOutput=fsr31\r\n[NvApi]\r\nOverrideNvapiDll=true\r\n'
                        b'[Menu]\r\nShortcutKey=45\r\n')
            plugin = self.fixture(game, original)
            result = self.setup(game, plugin)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(next(game.glob('OptiScaler.ini.d4r-backup.*')).read_bytes(), original)
            configured = (game / 'OptiScaler.ini').read_bytes()
            self.assertEqual(configured.count(b'\n'), configured.count(b'\r\n'))
            self.assertIn(b'LoadReShade=true\r\n', configured)
            self.assertIn(b'[FrameGen]\r\nFGOutput=fsr31\r\n', configured)
            self.assertIn(b'OverrideNvapiDll=true\r\n', configured)
            self.assertIn(b'; keep this comment\r\n', configured)
            ini = configparser.ConfigParser()
            ini.read_string(configured.decode())
            self.assertEqual(ini['Libraries']['NvngxPath'], self.winpath(plugin))
            self.assertEqual(ini['Plugins']['Path'], self.winpath(plugin.parent))
            self.assertEqual((game / 'winmm.dll').read_bytes(), b'existing OptiScaler loader')
            self.assertEqual((plugin.parent / 'Other.asi').read_bytes(), b'other mod')
            self.assertEqual((game / 'D3D12Core.dll').read_bytes(), b'original Microsoft SDK core')
            for name in ('d3d12.dll', 'd4r12core.dll'):
                self.assertEqual((game / name).read_bytes(), ('new ' + name).encode())
                self.assertEqual(next(game.glob(name + '.d4r-backup.*')).read_bytes(), ('old ' + name).encode())
            self.assertEqual(self.setup(game, plugin).returncode, 0)
            self.assertEqual(len(list(game.glob('*.d4r-backup.*'))), 3)
            self.assertEqual((game / 'OptiScaler.ini').read_bytes(), configured)
            check = subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), str(game)],
                                   text=True, capture_output=True)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            self.assertIn('(d4r ASI)', check.stdout)
            for old, new, message in ((b'LoadAsiPlugins=true', b'LoadAsiPlugins=false', 'LoadAsiPlugins=true'),
                                      (b'HookOriginalNvngxOnly=true', b'HookOriginalNvngxOnly=false',
                                       'HookOriginalNvngxOnly=true')):
                (game / 'OptiScaler.ini').write_bytes(configured.replace(old, new))
                check = subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), str(game)],
                                       text=True, capture_output=True)
                self.assertNotEqual(check.returncode, 0)
                self.assertIn(message, check.stdout)

    def test_duplicate_settings_and_missing_nvidia_leave_config_untouched(self):
        for source in (b'[Libraries]\nNvngxPath=auto\nnvngxpath=other\n',
                       b'[Plugins]\nPath=plugins\n[plugins]\nPath=elsewhere\n',
                       b'[Plugins]\nLoadAsiPlugins=true\n'):
            with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                game = Path(tmp)
                plugin = self.fixture(game, source)
                if source == b'[Plugins]\nLoadAsiPlugins=true\n':
                    (plugin.parent / 'd4r/nvngx_dlss.dll').unlink()
                result = self.setup(game, plugin)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((game / 'OptiScaler.ini').read_bytes(), source)
                self.assertFalse(list(game.glob('*.d4r-backup.*')))
                self.assertFalse(list(game.glob('.d4r-setup.*')))

    def test_saved_overlay_settings_survive_later_setup_and_runtime_update(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            plugin = self.fixture(game, b'[Plugins]\nLoadAsiPlugins=true\n')
            self.assertEqual(self.setup(game, plugin).returncode, 0)
            ini = game / 'OptiScaler.ini'
            saved = ini.read_bytes().replace(b'Dx12Upscaler=dlss', b'Dx12Upscaler=ffx')
            saved = saved.replace(b'Dxgi=true', b'Dxgi=false').replace(b'Enabled=true', b'Enabled=auto')
            saved += b'[FrameGen]\nEnabled=true\nFGInput=dlssg\nFGOutput=xefg\n[Menu]\nScale=1.25\n'
            ini.write_bytes(saved)
            # Normal relaunch must be a byte-for-byte no-op, including auto values.
            self.assertEqual(self.setup(game, plugin).returncode, 0)
            self.assertEqual(ini.read_bytes(), saved)
            # A runtime update can replace its DLLs without resetting overlay choices.
            (plugin.parent / 'd4r/d4r12core.dll').write_bytes(b'updated core')
            self.assertEqual(self.setup(game, plugin).returncode, 0)
            self.assertEqual((game / 'd4r12core.dll').read_bytes(), b'updated core')
            self.assertEqual(ini.read_bytes(), saved)

    def test_first_setup_keeps_existing_upscaler_and_spoofing_preferences(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            original = (b'[Upscalers]\nDx12Upscaler=ffx\n[DLSS]\nEnabled=auto\n'
                        b'[Spoofing]\nDxgi=false\n[FrameGen]\nFGOutput=fsrfg\n')
            plugin = self.fixture(game, original)
            self.assertEqual(self.setup(game, plugin).returncode, 0)
            saved = (game / 'OptiScaler.ini').read_bytes()
            for line in (b'Dx12Upscaler=ffx', b'Enabled=auto', b'Dxgi=false', b'FGOutput=fsrfg'):
                self.assertIn(line, saved)
            self.assertIn(self.winpath(plugin).encode(), saved)

    def test_relocated_plugin_refreshes_only_its_paths(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            plugin = self.fixture(game, b'[Plugins]\nLoadAsiPlugins=true\n')
            self.assertEqual(self.setup(game, plugin).returncode, 0)
            ini = game / 'OptiScaler.ini'
            saved = ini.read_bytes().replace(b'Dx12Upscaler=dlss', b'Dx12Upscaler=ffx')
            saved += b'[FrameGen]\nFGOutput=xefg\n'
            ini.write_bytes(saved)
            destination = game / 'moved plugins'
            shutil.move(str(plugin.parent), destination)
            moved = destination / plugin.name
            self.assertEqual(self.setup(game, moved).returncode, 0)
            expected = saved.replace(self.winpath(plugin.parent).encode(), self.winpath(destination).encode())
            self.assertEqual(ini.read_bytes(), expected)

    @unittest.skipUnless((ROOT / 'build/d4r.asi').exists(), 'Build the ASI first')
    def test_actual_asi_load_and_automatic_configuration(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp) / 'Game with spaces'
            original = b'[Plugins]\nLoadAsiPlugins=true\nPath=custom plugins\n'
            plugin = self.fixture(game, original)
            shutil.copy2(ROOT / 'build/d4r.asi', plugin)
            shutil.copy2(self.probe, game / 'probe.exe')
            result = subprocess.run(['wine', str(game / 'probe.exe'), self.winpath(plugin),
                                     self.winpath(plugin.parent / 'd4r')], env=self.env, cwd=self.directory,
                                    text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ASI loader contract passed', result.stdout)
            deadline = time.monotonic() + 15
            status = plugin.parent / 'd4r/installation-status.txt'
            while (not status.exists() or 'setup complete' not in status.read_text()) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIn(self.winpath(plugin), (game / 'OptiScaler.ini').read_text())
            log = (plugin.parent / 'd4r/d4r_nvngx.log').read_text()
            self.assertEqual(log.count('InitializeASI:'), 1)
            self.assertFalse((plugin.parent / 'd4r_nvngx.log').exists())

    def test_invalid_second_dll_destination_preserves_first_dll_and_config(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            source = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, source)
            (game / 'd4r12core.dll').unlink()
            (game / 'd4r12core.dll').mkdir()
            result = self.setup(game, plugin)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((game / 'd3d12.dll').read_bytes(), b'old d3d12.dll')
            self.assertEqual((game / 'OptiScaler.ini').read_bytes(), source)
            self.assertFalse(list(game.glob('.d4r-setup.*')))

    def test_nvidia_installer_accepts_both_runtime_layouts(self):
        for nested in (False, True):
            with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
                base = Path(tmp)
                runtime = base / 'd4r'
                (runtime / 'ngx').mkdir(parents=True)
                for name in ('nvngx.dll', 'd4r.ini'):
                    (runtime / name).touch()
                dlss, dlssd, ngx = b'verified DLSS fixture', b'verified DLSS RR fixture', b'verified NGX fixture'
                (runtime / 'nvngx_dlss.dll').write_bytes(dlss)
                (runtime / 'nvngx_dlssd.dll').write_bytes(dlssd)
                (runtime / 'ngx/_nvngx.dll').write_bytes(ngx)
                script = (ROOT / 'install.sh').read_text()
                replacements = {'DLSS_SHA256': hashlib.sha256(dlss).hexdigest(),
                                'DLSSD_SHA256': hashlib.sha256(dlssd).hexdigest(),
                                'NGX_SHA256': hashlib.sha256(ngx).hexdigest()}
                script = '\n'.join(key + '=' + replacements[key] if key in replacements else line
                                   for line in script.splitlines() for key in [line.split('=', 1)[0]]) + '\n'
                installer = (runtime if nested else base) / 'install.sh'
                installer.write_text(script)
                result = subprocess.run(['bash', str(installer)], cwd=self.directory,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('already installed and verified', result.stdout)
                self.assertEqual((runtime / 'nvngx_dlss.dll').read_bytes(), dlss)

    def test_locked_second_dll_rolls_back_first_dll(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            original = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, original)
            result = subprocess.run(['wine', str(self.lock_probe), self.winpath(self.helper),
                                     self.winpath(game / 'd4r12core.dll'), self.winpath(game), self.winpath(plugin)],
                                    env=self.env, cwd=self.directory, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ('d3d12.dll', 'd4r12core.dll'):
                self.assertEqual((game / name).read_bytes(), ('old ' + name).encode())
            self.assertEqual((game / 'OptiScaler.ini').read_bytes(), original)
            self.assertIn('earlier changes were restored', (plugin.parent / 'd4r/installation-status.txt').read_text())

    def test_d3d12_optiscaler_loader_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            original = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, original)
            loader = b'MZ fixture OptiScaler.dll\0'
            (game / 'd3d12.dll').write_bytes(loader)
            result = self.setup(game, plugin)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((game / 'd3d12.dll').read_bytes(), loader)
            self.assertEqual((game / 'OptiScaler.ini').read_bytes(), original)
            self.assertFalse(list(game.glob('*.d4r-backup.*')))

    def test_prefix_config_detected_without_game_local_ini(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            base = Path(tmp)
            game = base / 'game'
            original = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, original)
            umu = base / 'prefix/drive_c/windows/system32/umu'
            umu.mkdir(parents=True)
            shutil.move(str(plugin.parent), umu / 'plugins')
            plugin = umu / 'plugins/d4r.asi'
            config = umu / 'OptiScaler.ini'
            shutil.move(str(game / 'OptiScaler.ini'), config)
            result = self.setup(game, plugin)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(self.winpath(plugin), config.read_text())
            self.assertFalse((game / 'OptiScaler.ini').exists())
            self.assertEqual(next(umu.glob('OptiScaler.ini.d4r-backup.*')).read_bytes(), original)
            # The real GE-Proton loader writes C: paths and lives in the prefix,
            # while the game EXE lives outside that prefix on Z:.
            dosdevices = base / 'prefix/dosdevices'
            dosdevices.mkdir()
            (dosdevices / 'c:').symlink_to('../drive_c')
            (umu / 'OptiScaler.dll').touch()
            win_plugins = r'C:\windows\system32\umu\plugins'
            data = config.read_text().replace(self.winpath(plugin.parent), win_plugins)
            config.write_text(data)
            checker = plugin.parent / 'd4r/d4r-check.sh'
            shutil.copy2(ROOT / 'packaging/d4r-check.sh', checker)
            (game / 'winmm.dll').unlink()
            result = subprocess.run(['sh', str(checker), str(game)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('(d4r ASI)', result.stdout)

    @unittest.skipUnless((ROOT / 'build/d4r.asi').exists(), 'Build the ASI first')
    def test_active_proxy_ini_selected_over_unrelated_game_ini(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            base = Path(tmp)
            game = base / 'game'
            original = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, original)
            shutil.copy2(ROOT / 'build/d4r.asi', plugin)
            umu = base / 'prefix/drive_c/windows/system32/umu'
            umu.mkdir(parents=True)
            config = umu / 'OptiScaler.ini'
            config.write_bytes(b'; active proxy config\n[Plugins]\nLoadAsiPlugins=true\n')
            proxy_source = base / 'proxy.cpp'
            proxy_source.write_text('extern "C" __declspec(dllexport) void DummyOpti() {}\n')
            definition = base / 'proxy.def'
            definition.write_text('LIBRARY OptiScaler.dll\nEXPORTS\nDummyOpti\n')
            proxy = umu / 'winmm.dll'
            subprocess.run(['x86_64-w64-mingw32-g++', '-shared', '-static', str(proxy_source),
                            str(definition), '-o', str(proxy)], check=True, capture_output=True)
            source = base / 'active-proxy.cpp'
            source.write_text(r'''
#include <windows.h>
int wmain(int argc, wchar_t** argv) {
    if (argc != 3 || !LoadLibraryW(argv[1])) return 2;
    HMODULE asi = LoadLibraryW(argv[2]);
    if (!asi) return 3;
    auto init = reinterpret_cast<void (*)()>(GetProcAddress(asi, "InitializeASI"));
    if (!init) return 4;
    init();
    return 0;
}
''')
            probe = game / 'active-proxy.exe'
            subprocess.run(['x86_64-w64-mingw32-g++', '-static', '-municode', str(source), '-o', str(probe)],
                           check=True, capture_output=True)
            result = subprocess.run(['wine', str(probe), self.winpath(proxy), self.winpath(plugin)],
                                    env=self.env, cwd=self.directory, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            status = plugin.parent / 'd4r/installation-status.txt'
            deadline = time.monotonic() + 15
            while (not status.exists() or 'setup complete' not in status.read_text()) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(status.exists())
            self.assertIn('setup complete', status.read_text())
            self.assertIn(self.winpath(plugin), config.read_text())
            self.assertEqual((game / 'OptiScaler.ini').read_bytes(), original)

    @unittest.skipUnless((ROOT / 'build/d4r.asi').exists(), 'Build the ASI first')
    def test_xalia_does_not_start_setup_or_modify_settings(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            original = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, original)
            shutil.copy2(ROOT / 'build/d4r.asi', plugin)
            utility = game / 'xalia.exe'
            shutil.copy2(self.probe, utility)
            result = subprocess.run(['wine', str(utility), self.winpath(plugin),
                                     self.winpath(plugin.parent / 'd4r')], env=self.env, cwd=self.directory,
                                    text=True, capture_output=True, timeout=30)
            # The probe's environment assertions intentionally fail because
            # InitializeASI must leave this auxiliary process untouched.
            self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
            self.assertEqual((game / 'OptiScaler.ini').read_bytes(), original)
            self.assertFalse((plugin.parent / 'd4r/installation-status.txt').exists())
            self.assertFalse((plugin.parent / 'd4r/d4r_nvngx.log').exists())

    @unittest.skipUnless((ROOT / 'build/d4r.asi').exists(), 'Build the ASI first')
    def test_installation_waits_for_game_exit(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            game = Path(tmp)
            original = b'[Plugins]\nLoadAsiPlugins=true\n'
            plugin = self.fixture(game, original)
            shutil.copy2(ROOT / 'build/d4r.asi', plugin)
            shutil.copy2(self.probe, game / 'probe.exe')
            process = subprocess.Popen(['wine', str(game / 'probe.exe'), self.winpath(plugin),
                                        self.winpath(plugin.parent / 'd4r')],
                                       env=dict(self.env, D4R_ASI_PROBE_PAUSE_MS='5000'), cwd=self.directory,
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                status = plugin.parent / 'd4r/installation-status.txt'
                deadline = time.monotonic() + 15
                while not status.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(status.exists())
                self.assertIn('Close the game', status.read_text())
                self.assertEqual((game / 'd3d12.dll').read_bytes(), b'old d3d12.dll')
                self.assertEqual((game / 'OptiScaler.ini').read_bytes(), original)
                output, errors = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, output + errors)
                deadline = time.monotonic() + 10
                while 'setup complete' not in status.read_text() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIn('setup complete', status.read_text())
                self.assertEqual((game / 'd3d12.dll').read_bytes(), b'new d3d12.dll')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == '__main__':
    unittest.main()
