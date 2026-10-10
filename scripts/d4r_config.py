#!/usr/bin/env python3
"""Translate d4r.ini (see config/d4r.ini.default) into shell exports for the game launcher.

usage: d4r_config.py [--config PATH] [--print | --launch]
  Prints "export VAR=value" lines for every setting the INI decides, skipping variables that are
  already set in the environment (explicit environment always wins). --print lists the resulting
  settings for humans instead. Without --config: $D4R_CONFIG, else ~/.config/d4r/d4r.ini, which is
  created from config/d4r.ini.default on first use.
"""
import configparser
import os
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = ROOT / "config" / "d4r.ini.default"
PRESETS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "J": 10, "K": 11, "L": 12, "M": 13}
# friendly names for [DLSS] Model
MODEL_ALIASES = {"CNN": "E", "DLSS3": "E", "DLSS4": "K", "DLSS4.5": "M", "TRANSFORMER": "K"}


def fail(message):
    print(f"d4r config: {message}", file=sys.stderr)
    sys.exit(2)


def config_path(argv):
    if "--config" in argv:
        return Path(argv[argv.index("--config") + 1]).expanduser()
    if os.environ.get("D4R_CONFIG"):
        return Path(os.environ["D4R_CONFIG"]).expanduser()
    path = Path.home() / ".config" / "d4r" / "d4r.ini"
    if not path.exists() and DEFAULT_TEMPLATE.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_TEMPLATE, path)
        print(f"d4r config: created {path} from the default template; edit it to change settings",
              file=sys.stderr)
    return path


def main(argv):
    path = config_path(argv)
    ini = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    ini.optionxform = str  # keep [Env] variable names as written
    if path.exists():
        ini.read(path, encoding="utf-8-sig")

    def get(section, key):
        value = ini.get(section, key, fallback="").strip()
        return None if value == "" or value.lower() == "auto" else value

    def flag(section, key):
        value = get(section, key)
        if value is None:
            return None
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        fail(f"[{section}] {key} must be true or false, not {value!r}")

    def path_value(section, key):
        value = get(section, key)
        return os.path.expanduser(value) if value else None

    if "--launch" in argv:
        # PROTON_DIR, COMPAT_DATA_DIR and GAME_EXE for scripts/d4r_play.sh, one per line
        values = [path_value("Launch", key) for key in ("ProtonDir", "CompatDataDir", "GameExe")]
        missing = [key for key, value in zip(("ProtonDir", "CompatDataDir", "GameExe"), values) if not value]
        if missing:
            fail(f"set [Launch] {', '.join(missing)} in {path}")
        print("\n".join(values))
        return

    env = {}
    env["D4R_PREFER_ACCURACY"] = "1" if flag("Kernels", "PreferAccuracy") else "0"

    model = get("DLSS", "Model")
    if model is not None:
        if model.isdigit():
            env["D4R_DLSS_PRESET"] = model
        elif model.upper() in PRESETS or model.upper() in MODEL_ALIASES:
            env["D4R_DLSS_PRESET"] = str(PRESETS[MODEL_ALIASES.get(model.upper(), model.upper())])
        else:
            fail(f"[DLSS] Model {model!r}: use one of {', '.join(list(PRESETS) + list(MODEL_ALIASES))} or a preset number")

    age = get("Latency", "FrameAge")
    if age is not None:
        if age not in ("0", "1", "2", "3"):
            fail(f"[Latency] FrameAge must be 0-3, not {age!r}")
        if age == "0":
            # same-frame results: the game's command list is split around DLSS (patched vkd3d-proton)
            env["D4R_SHIM_SPLIT_FRAME"] = "1"
        else:
            env["D4R_SHIM_SPLIT_FRAME"] = "0"
            env["D4R_SHIM_MAX_IN_FLIGHT"] = age

    for key, var in (("RuntimeDir", "D4R_RUNTIME_DIR"), ("ZludaDir", "D4R_ZLUDA_DIR"), ("RocmDir", "D4R_ROCM_DIR"),
                     ("VkD3DDir", "D4R_VKD3D_DIR")):
        value = path_value("Paths", key)
        if value:
            env[var] = value
    # JIT caches (ZLUDA's compiled kernels among them): a directory, or per-zluda for the one the test
    # runs warm for each ZLUDA build (~/.cache/d4r-perf-game-jit-<ZludaDir name>)
    jit = get("Paths", "JitCacheDir")
    if jit is not None:
        if jit.lower() == "per-zluda":
            zluda = env.get("D4R_ZLUDA_DIR") or os.environ.get("D4R_ZLUDA_DIR")
            if not zluda:
                fail("[Paths] JitCacheDir = per-zluda needs [Paths] ZludaDir")
            jit = f"~/.cache/d4r-perf-game-jit-{Path(zluda).name}"
        env["XDG_CACHE_HOME"] = os.path.expanduser(jit)
    if age == "0" and "D4R_VKD3D_DIR" not in env and not os.environ.get("D4R_VKD3D_DIR"):
        print("d4r config: FrameAge = 0 needs [Paths] VkD3DDir (the d4r-patched vkd3d-proton); without it the "
              "shim falls back to older frames", file=sys.stderr)

    mode = (get("Kernels", "NativeKernels") or "off").lower()
    if mode not in ("fast", "exact", "off"):
        fail(f"[Kernels] NativeKernels must be fast, exact or off, not {mode!r}")
    if mode != "off":
        directory = path_value("Paths", "NativeKernelDirFast" if mode == "fast" else "NativeKernelDirExact")
        if not directory:
            fail(f"[Kernels] NativeKernels = {mode} needs [Paths] NativeKernelDir{mode.capitalize()}")
        if not Path(directory).is_dir():
            fail(f"native kernel directory {directory} does not exist")
        env["D4R_ZLUDA_NATIVE_DIR"] = directory
    for key, var in (("Fp8Wmma", "D4R_ZLUDA_WMMA_FP8"), ("IgnoreDenormals", "D4R_ZLUDA_IGNORE_DENORMAL")):
        value = flag("Kernels", key)
        if value is not None:
            env[var] = "1" if value else "0"
    # RDNA4's native FP8 WMMA (on unless set off; ZLUDA and the bridge ignore it on other GPUs)
    value = flag("Kernels", "NativeFp8")
    env["D4R_ZLUDA_WMMA_FP8_NATIVE"] = "0" if value is False else "1"
    value = flag("Kernels", "NativeSwinEncoders")
    env["D4R_NATIVE_SWIN_ENCODERS"] = "0" if value is False else "1"
    value = get("Kernels", "ImplicitMaxBlock")
    if value:
        env["D4R_ZLUDA_IMPLICIT_MAX_BLOCK"] = value

    for key, var in (("VramInterop", "D4R_SHIM_VRAM_INTEROP"), ("InputSync", "D4R_SHIM_INPUT_SYNC"),
                     ("LinearInputs", "D4R_SHIM_LINEAR_INPUTS"),
                     ("DirectOutput", "D4R_SHIM_OUTPUT_DIRECT"), ("ElideNgxSync", "D4R_ELIDE_NGX_SYNC"),
                     ("EvalSync", "D4R_SHIM_EVAL_SYNC")):
        value = flag("Interop", key)
        if value is not None:
            env[var] = "1" if value else "0"
    value = get("Interop", "MarkerPollUs") or "200"
    dilation = get("Interop", "MotionVectorDilation")
    if dilation is not None and dilation not in ("0", "1", "2"):
        fail("[Interop] MotionVectorDilation must be 0, 1 or 2")
    env["D4R_MOTION_DILATION"] = dilation or "0"
    if value:
        if not value.isdigit():
            fail(f"[Interop] MarkerPollUs must be a number of microseconds, not {value!r}")
        env["D4R_SHIM_MARKER_POLL_US"] = value

    value = flag("Engine", "Enabled")
    env["D4R_ENGINE"] = "1" if value else "0"
    if path_value("Engine", "ModelDir"):
        env["D4R_ENGINE_MODEL_DIR"] = path_value("Engine", "ModelDir")
    network = get("Engine", "Network")
    if network is not None and network not in ("auto", "hip", "vulkan"):
        fail("[Engine] Network must be auto, hip or vulkan")
    if network in ("hip", "vulkan"):
        env["D4R_ENGINE_NETWORK"] = network

    engine = path_value("Game", "EngineIni")
    cvars = get("Game", "Cvars")
    if engine and cvars:
        env["D4R_UE_ENGINE_INI"] = engine
        env["D4R_UE_CVARS"] = cvars

    value = flag("OptiScaler", "FpsOverlay")
    if value is False:
        env["D4R_OPTISCALER_FPS_OVERLAY_TYPE"] = "0"
        env["D4R_OPTISCALER_SHOW_FPS"] = "false"
    elif get("OptiScaler", "FpsOverlayType"):
        env["D4R_OPTISCALER_FPS_OVERLAY_TYPE"] = get("OptiScaler", "FpsOverlayType")
    if get("OptiScaler", "LogLevel"):
        env["D4R_OPTISCALER_LOG_LEVEL"] = get("OptiScaler", "LogLevel")
    extra = []
    ratios = [(mode_name, get("OptiScaler", f"QualityRatio{mode_name}"))
              for mode_name in ("Quality", "Balanced", "Performance", "UltraPerformance")]
    if any(ratio for _, ratio in ratios):
        extra.append("QualityOverrides.QualityRatioOverrideEnabled=true")
        extra += [f"QualityOverrides.QualityRatio{name}={ratio}" for name, ratio in ratios if ratio]
    if get("OptiScaler", "Extra"):
        extra.append(get("OptiScaler", "Extra"))
    if extra:
        env["D4R_EXTRA_OPTISCALER_CONFIG"] = ";".join(extra)

    if flag("Overlay", "MangoHud"):
        env["MANGOHUD"] = "1"
        if get("Overlay", "MangoHudConfig"):
            env["MANGOHUD_CONFIG"] = get("Overlay", "MangoHudConfig")

    if flag("Debug", "Profile"):
        env["D4R_PROFILE"] = "1"
    if path_value("Debug", "LogDir"):
        env["D4R_GAME_LOG_DIR"] = path_value("Debug", "LogDir")

    if ini.has_section("Env"):
        for key, value in ini.items("Env"):
            if not (key.isascii() and key.isidentifier()):
                fail(f"[Env] key {key!r} must be an ASCII identifier")
            env[key] = value.strip()

    final = {key: value for key, value in env.items() if key not in os.environ}
    if "--print" in argv:
        print(f"d4r config: {path}{'' if path.exists() else ' (missing: built-in defaults)'}")
        for key in sorted(env):
            note = "" if key in final else f"   (environment wins: {os.environ[key]})"
            print(f"  {key}={env[key]}{note}")
        return
    for key, value in sorted(final.items()):
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main(sys.argv[1:])
