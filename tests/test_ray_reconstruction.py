"""Preset isolation, invalid-setting rejection, and environment precedence."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "d4r_config.py"
RR_VARIABLES = ("D4R_RR_ENABLE", "D4R_RR_PRESET", "D4R_RR_BACKEND", "D4R_FSR_RR_DLL", "D4R_FSR_RR_STABILITY_BIAS",
                "D4R_FSR_RR_DETAIL")


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

    def test_dlss_45_ray_reconstruction_is_model_f(self):
        code, decided = run_config("[RayReconstruction]\nEnable = true\nModel = F\n")
        self.assertEqual(code, 0)
        self.assertEqual(decided["D4R_RR_PRESET"], "6")     # preset F: nvngx_dlssd.dll 310.9 or later

    def test_invalid_rr_settings_are_rejected_rather_than_guessed(self):
        for text in ("[RayReconstruction]\nModel = G\n",
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


class RayReconstructionBackendTests(unittest.TestCase):
    """[RayReconstruction] Backend picks the denoiser; it never changes the DLSS presets."""

    def test_default_leaves_dlss_in_place(self):
        code, decided = run_config("[RayReconstruction]\nEnable = true\n")
        self.assertEqual(code, 0)
        self.assertNotIn("D4R_RR_BACKEND", decided)  # the shim's default is dlss

    def test_backends_are_exported_and_independent_of_presets(self):
        for backend in ("dlss", "fsr", "FSR"):
            with self.subTest(backend=backend):
                code, decided = run_config(f"[RayReconstruction]\nBackend = {backend}\nModel = E\n[DLSS]\nModel = K\n")
                self.assertEqual(code, 0)
                self.assertEqual(decided["D4R_RR_BACKEND"], backend.lower())
                self.assertEqual(decided["D4R_RR_PRESET"], "5")
                self.assertEqual(decided["D4R_DLSS_PRESET"], "11")

    def test_unknown_backend_is_rejected(self):
        for value in ("amd", "fsr4", "native"):
            with self.subTest(value=value):
                code, decided = run_config(f"[RayReconstruction]\nBackend = {value}\n")
                self.assertNotEqual(code, 0)
                self.assertNotIn("D4R_RR_BACKEND", decided)

    def test_denoiser_path_and_environment_precedence(self):
        code, decided = run_config("[RayReconstruction]\nBackend = fsr\nFsrDenoiser = /opt/amd/denoiser.dll\n")
        self.assertEqual(code, 0)
        self.assertEqual(decided["D4R_FSR_RR_DLL"], "/opt/amd/denoiser.dll")
        code, decided = run_config("[RayReconstruction]\nBackend = fsr\n", D4R_RR_BACKEND="dlss")
        self.assertEqual(code, 0)
        self.assertNotIn("D4R_RR_BACKEND", decided)

    def test_templates_keep_dlss_as_the_default(self):
        import configparser
        for name in ("config/d4r.ini.default", "packaging/d4r.ini"):
            ini = configparser.ConfigParser(inline_comment_prefixes=(";",))
            ini.read(ROOT / name)
            self.assertEqual(ini.get("RayReconstruction", "Backend"), "dlss", name)
            self.assertEqual(ini.get("RayReconstruction", "FsrDenoiser"), "auto", name)
            self.assertEqual(ini.get("RayReconstruction", "FsrStabilityBias"), "auto", name)
            self.assertEqual(ini.get("RayReconstruction", "FsrDetail"), "0", name)

    def test_fsr_tuning_keys(self):
        code, decided = run_config("[RayReconstruction]\nBackend = fsr\nFsrStabilityBias = auto\nFsrDetail = 0.25\n")
        self.assertEqual(code, 0)
        self.assertNotIn("D4R_FSR_RR_STABILITY_BIAS", decided)  # auto: AMD's own default
        self.assertEqual(decided["D4R_FSR_RR_DETAIL"], "0.25")
        code, decided = run_config("[RayReconstruction]\nFsrStabilityBias = 0\n")
        self.assertEqual(code, 0)
        self.assertEqual(decided["D4R_FSR_RR_STABILITY_BIAS"], "0")
        for text in ("FsrStabilityBias = -1", "FsrStabilityBias = low", "FsrDetail = 1.5", "FsrDetail = x"):
            with self.subTest(text=text):
                code, _ = run_config(f"[RayReconstruction]\n{text}\n")
                self.assertNotEqual(code, 0)
        code, decided = run_config("[RayReconstruction]\nFsrStabilityBias = 0\n", D4R_FSR_RR_STABILITY_BIAS="1")
        self.assertEqual(code, 0)
        self.assertNotIn("D4R_FSR_RR_STABILITY_BIAS", decided)


if __name__ == "__main__":
    unittest.main()