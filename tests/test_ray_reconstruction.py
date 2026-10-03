"""Preset isolation, invalid-setting rejection, and environment precedence."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "d4r_config.py"
RR_VARIABLES = ("D4R_RR_ENABLE", "D4R_RR_PRESET")


def run_config(text, **overrides):
    """Runs d4r_config.py over `text`; returns (exit code, exports of interest).

    d4r_config.py also emits its own defaults for unrelated variables, so the result is
    filtered to the Ray Reconstruction and Super Sampling preset variables - the ones whose
    independence from each other is under test here.
    """
    with tempfile.TemporaryDirectory() as directory:
        ini = Path(directory) / "d4r.ini"
        ini.write_text(text)
        env = dict(os.environ)
        for name in RR_VARIABLES + ("D4R_DLSS_PRESET",):
            env.pop(name, None)
        env.update(overrides)
        result = subprocess.run(["python3", str(CONFIG), "--config", str(ini)], env=env,
                                capture_output=True, text=True)
    interesting = set(RR_VARIABLES) | {"D4R_DLSS_PRESET"}
    decided = {}
    for line in result.stdout.splitlines():
        if line.startswith("export "):
            key, _, value = line[len("export "):].partition("=")
            if key in interesting:
                decided[key] = value
    return result.returncode, decided


class RayReconstructionConfigTests(unittest.TestCase):
    def test_rr_and_super_sampling_presets_do_not_collide(self):
        code, decided = run_config("[RayReconstruction]\nEnable = true\nModel = E\n[DLSS]\nModel = K\n")
        self.assertEqual(code, 0)
        self.assertEqual(decided["D4R_RR_ENABLE"], "1")
        self.assertEqual(decided["D4R_RR_PRESET"], "5")     # Ray Reconstruction preset E
        self.assertEqual(decided["D4R_DLSS_PRESET"], "11")  # Super Sampling preset K

    def test_invalid_rr_settings_are_rejected_rather_than_guessed(self):
        for text in ("[RayReconstruction]\nModel = F\n",
                     "[RayReconstruction]\nModel = K\n",
                     "[RayReconstruction]\nEnable = maybe\n"):
            with self.subTest(ini=text):
                code, decided = run_config(text)
                self.assertNotEqual(code, 0, f"accepted {text!r} as {decided}")
                for name in RR_VARIABLES:
                    self.assertNotIn(name, decided)

    def test_environment_wins_over_the_ini(self):
        # An explicitly exported variable is the user's decision and must survive.
        code, decided = run_config("[RayReconstruction]\nEnable = true\nModel = E\n",
                                   D4R_RR_ENABLE="0", D4R_RR_PRESET="4")
        self.assertEqual(code, 0)
        self.assertNotIn("D4R_RR_ENABLE", decided)
        self.assertNotIn("D4R_RR_PRESET", decided)


if __name__ == "__main__":
    unittest.main()