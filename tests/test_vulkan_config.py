"""A native Vulkan install must leave d4r's CUDA core load outside OptiScaler's hook."""
from pathlib import Path
import configparser
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / 'packaging/optiscaler-vulkan.settings'


class VulkanConfigTests(unittest.TestCase):
    def test_migrate_broad_hook_preserving_game_specific_choices(self):
        sections = {}
        for line in SETTINGS.read_text().splitlines():
            if line and not line.startswith('#'):
                name, value = line.split('=', 1)
                section, key = name.split('.', 1)
                sections.setdefault(section, {})[key] = 'auto'
        sections['Hooks']['HookOriginalNvngxOnly'] = 'false'
        sections['Spoofing'] = {'Vulkan': 'false', 'VulkanExtensionSpoofing': 'false'}
        sections['Plugins'] = {'Path': r'd4r\plugins', 'LoadAsiPlugins': 'true'}
        source = '; existing game setup\r\n' + ''.join(
            f'[{name}]\r\n' + ''.join(f'{key}={value}\r\n' for key, value in values.items())
            for name, values in sections.items())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            incoming, outgoing = folder / 'old.ini', folder / 'new.ini'
            incoming.write_bytes(source.encode())
            subprocess.run(['python3', str(ROOT / 'scripts/configure_optiscaler.py'),
                            str(incoming), str(SETTINGS), str(outgoing)], check=True)
            data = outgoing.read_bytes()
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read_string(data.decode())
            self.assertEqual(cfg['Hooks']['HookOriginalNvngxOnly'], 'true')
            self.assertEqual(cfg['Libraries']['NvngxPath'], r'd4r\nvngx.dll')
            self.assertEqual(cfg['Upscalers']['VulkanUpscaler'], 'dlss')
            self.assertEqual(cfg['FrameGen']['Enabled'], 'false')
            self.assertEqual(cfg['Spoofing']['Vulkan'], 'false')
            self.assertEqual(cfg['Spoofing']['VulkanExtensionSpoofing'], 'false')
            self.assertEqual(cfg['Plugins']['Path'], r'd4r\plugins')
            self.assertEqual(cfg['Plugins']['LoadAsiPlugins'], 'true')
            self.assertEqual(data.count(b'\r\n'), data.count(b'\n'))
            self.assertTrue(data.startswith(b'; existing game setup\r\n'))


if __name__ == '__main__':
    unittest.main()
