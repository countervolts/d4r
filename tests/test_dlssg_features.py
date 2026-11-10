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


if __name__ == '__main__':
    unittest.main()
