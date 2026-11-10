"""d4r-features.py --fg dlssg: OptiScaler routing through the DLSS-G host and back, d4r.ini, the prefix's HAGS key."""
import configparser
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('d4r_features', ROOT / 'packaging/d4r-features.py')
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

OPTISCALER_INI = """[Upscalers]\r\nDx12Upscaler=dlss\r\n[FrameGen]\r\nEnabled=false\r\nFGInput=nofg\r\nFGOutput=nofg\r\n\
[Libraries]\r\nOptiDllPath=d4r\r\nNvngxPath=d4r\\nvngx.dll\r\n[Spoofing]\r\nDxgi=auto\r\n[fakenvapi]\r\nUseFakenvapi=false\r\n"""


def read(path):
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(path.read_text())
    return parsed


class DlssgFeatureTests(unittest.TestCase):
    def test_route_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            game = Path(directory)
            ini = game / 'OptiScaler.ini'
            ini.write_bytes(OPTISCALER_INI.encode())
            (game / 'd4r').mkdir()
            backup = features.configure(ini, game / 'd4r', 'dlssg', None, 'auto')
            self.assertTrue(backup and backup.read_bytes() == OPTISCALER_INI.encode())
            cfg = read(ini)
            self.assertEqual(cfg['Libraries']['NvngxPath'], 'd4r\\fg\\dlssg\\nvngx.dll_dlssg_host.dll')
            self.assertEqual(cfg['Spoofing']['Dxgi'], 'true')
            self.assertEqual(cfg['Spoofing']['SpoofHAGS'], 'true')
            self.assertEqual(cfg['FrameGen']['Enabled'], 'false')
            self.assertIn(b'\r\n', ini.read_bytes())  # line endings kept
            features.configure(ini, game / 'd4r', 'off', None, 'auto')
            self.assertEqual(read(ini)['Libraries']['NvngxPath'], 'd4r\\nvngx.dll')

    def test_d4r_ini_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / 'd4r.ini').write_text((ROOT / 'packaging/d4r.ini').read_text())
            features.set_d4r_ini(runtime, 'dlssg', False, Path('/home/u/.cache/nvxrt-gfx11'))
            cfg = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(';',))
            cfg.read(runtime / 'd4r.ini')
            self.assertEqual(cfg['FrameGeneration']['Backend'], 'dlssg')
            self.assertEqual(cfg['FrameGeneration']['Exact'], 'false')
            self.assertEqual(cfg['FrameGeneration']['KernelCache'], '/home/u/.cache/nvxrt-gfx11')
            self.assertEqual(cfg['FrameGeneration']['Model'], 'auto')
            features.set_d4r_ini(runtime, 'dlssg', False, Path('/c'), lite=True)
            cfg.read(runtime / 'd4r.ini')
            self.assertEqual(cfg['FrameGeneration']['Model'], 'lite')
            # a later setup (install.sh again) keeps the user's Model and Exact
            text = (runtime / 'd4r.ini').read_text().replace('Exact=false', 'Exact=true').replace('Exact = false', 'Exact = true')
            (runtime / 'd4r.ini').write_text(text)
            features.set_d4r_ini(runtime, 'dlssg', False, Path('/c'))
            cfg.read(runtime / 'd4r.ini')
            self.assertEqual(cfg['FrameGeneration']['Model'], 'lite')
            self.assertEqual(cfg['FrameGeneration']['Exact'], 'true')
            features.set_d4r_ini(runtime, 'off')
            cfg.read(runtime / 'd4r.ini')
            self.assertEqual(cfg['FrameGeneration']['Backend'], 'off')

    def test_hags_once(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            reg = prefix / 'system.reg'
            reg.write_text('WINE REGISTRY Version 2\n\n[System\\\\ControlSet001\\\\Control\\\\GraphicsDrivers] 1\n'
                           '#time=1\n"HwSchMode"=dword:00000001\n\n[Other] 1\n"x"="y"\n')
            self.assertTrue(features.set_hags(prefix))
            text = reg.read_text()
            self.assertIn('"HwSchMode"=dword:00000002', text)
            self.assertNotIn('"HwSchMode"=dword:00000001', text)
            self.assertIn('[Other] 1\n"x"="y"', text)
            self.assertFalse(features.set_hags(prefix))

    def test_install_root_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            steamapps = Path(directory) / 'steamapps'
            game = steamapps / 'common' / 'Game'
            (game / 'sub').mkdir(parents=True)
            (game / 'sub/nvngx_dlssg.dll').write_bytes(b'x')
            (steamapps / 'common/Other').mkdir()
            (steamapps / 'common/Other/nvngx_dlssg.dll').write_bytes(b'y')
            (steamapps / 'appmanifest_42.acf').write_text('"AppState"\n{\n\t"appid"\t\t"42"\n\t"installdir"\t\t"Game"\n}\n')
            (steamapps / 'compatdata/42/pfx').mkdir(parents=True)
            (steamapps / 'compatdata/42/pfx/system.reg').write_text('')
            self.assertEqual(features.find_dlssg(game), game / 'sub/nvngx_dlssg.dll')
            self.assertEqual(features.proton_prefix(game), steamapps / 'compatdata/42/pfx')
            self.assertEqual(features.proton_prefix(game / 'sub'), steamapps / 'compatdata/42/pfx')

    def test_optiscaler_dlssg_for_games_without_dlssg(self):
        with tempfile.TemporaryDirectory() as directory:
            steamapps = Path(directory) / 'steamapps'
            game = steamapps / 'common' / 'Game'
            (game / 'd4r').mkdir(parents=True)
            self.assertFalse(features.game_streamline(game))
            (game / 'Engine/Streamline').mkdir(parents=True)
            (game / 'Engine/Streamline/sl.dlss_g.dll').write_bytes(b'x')
            self.assertTrue(features.game_streamline(game))
            other = steamapps / 'common' / 'Other'
            (other / 'OptiScaler/streamline').mkdir(parents=True)
            (other / 'OptiScaler/streamline/sl.dlss_g.dll').write_bytes(b'x')   # ours from an earlier run: not the game's
            self.assertFalse(features.game_streamline(other))
            ini = other / 'OptiScaler.ini'
            ini.write_bytes(OPTISCALER_INI.replace('OptiDllPath=d4r', 'OptiDllPath=OptiScaler').encode())
            src = Path(directory) / 'sl'
            src.mkdir()
            for name in features.STREAMLINE_DLLS + ('nvngx_dlssg.dll',):
                (src / name).write_bytes(name.encode())
            folder = features.stage_streamline(ini, src)
            self.assertEqual(folder, other / 'OptiScaler/streamline')
            self.assertEqual((folder / 'nvngx_dlssg.dll').read_bytes(), b'nvngx_dlssg.dll')
            (other / 'd4r').mkdir()
            features.configure(ini, other / 'd4r', 'dlssg', None, 'auto', optifg=True)
            cfg = read(ini)
            self.assertEqual(cfg['FrameGen']['Enabled'], 'true')
            self.assertEqual(cfg['FrameGen']['FGInput'], 'upscaler')
            self.assertEqual(cfg['FrameGen']['FGOutput'], 'dlssg')
            self.assertEqual(cfg['Libraries']['NvngxPath'], 'd4r\\fg\\dlssg\\nvngx.dll_dlssg_host.dll')


if __name__ == '__main__':
    unittest.main()
