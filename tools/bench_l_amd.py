#!/usr/bin/env python3
"""Interleave L model variants and report timing plus RGB PSNR to supplied references.

Case JSON: render/output [W,H], frames [{color,motion,depth,jitter:[x,y],reference}],
optional exclude_rects [[x0,y0,x1,y1]]. Paths are relative to the case file.
References and outputs are raw RGBA16F. No clipping or nonfinite masking.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

import numpy as np


def quality(got, reference, width, height, exclude):
    a = np.fromfile(got, np.float16).reshape(height, width, 4)[..., :3].astype(np.float64)
    b = np.fromfile(reference, np.float16).reshape(height, width, 4)[..., :3].astype(np.float64)
    if not np.isfinite(b).all():
        raise ValueError(f"nonfinite reference: {reference}")
    if not np.isfinite(a).all():
        return {"nonfinite_rgb": int((~np.isfinite(a)).sum()), "psnr_unit_peak_db": None,
                "psnr_reference_peak_db": None}
    mask = np.ones((height, width), bool)
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = False
    if not mask.any():
        raise ValueError("quality mask excludes every pixel")
    a, b = a[mask], b[mask]
    mse = float(np.mean((a - b) ** 2))
    peak = max(float(np.max(np.abs(b))), 1e-6)
    # JSON null for an infinite PSNR; zero MSE records equality without using it
    # as an acceptance condition.
    return {"mse": mse, "reference_peak": peak, "nonfinite_rgb": 0,
            "psnr_unit_peak_db": -10 * math.log10(mse) if mse else None,
            "psnr_reference_peak_db": 10 * math.log10(peak * peak / mse) if mse else None,
            "max_abs_error": float(np.max(np.abs(a - b)))}


def run(args):
    if args.trials < 1 or args.warmup_passes < 0 or args.passes <= args.warmup_passes:
        raise ValueError("need trials > 0 and passes > warmup_passes >= 0")
    case = json.loads(args.case.read_text())
    resolve = lambda name: str((args.case.parent / name).resolve())
    frames = []
    for frame in case["frames"]:
        frames += [resolve(frame[k]) for k in ("color", "motion", "depth")]
        frames += list(map(str, frame["jitter"]))
    models = []
    for model in args.model:
        name, path = model.split("=", 1)
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in [n for n, _ in models]:
            raise ValueError("model labels must be unique simple names")
        models.append((name, str(Path(path).resolve())))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    env = dict(os.environ, D4R_M_WARMUP_PASSES=str(args.warmup_passes), D4R_M_TIME_EACH="1")
    for trial in range(args.trials):
        # Rotate and reverse so three or more variants do not always occupy
        # the same positions in the thermal/clock progression.
        order = models[trial % len(models):] + models[:trial % len(models)]
        if trial % 2:
            order = order[::-1]
        for name, model in order:
            prefix = (args.out / f"{name}-{trial}").resolve()
            cmd = [str(args.runner.resolve()), model, *map(str, case["render"]),
                   *map(str, case["output"]), str(prefix), str(args.passes), "vkhip", "--", *frames]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            log = result.stdout + result.stderr
            prefix.with_suffix(".log").write_text(log)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}): {log[-2000:]}")
            times = [float(x) for x in re.findall(r"^TIMED pass=\d+ frame=\d+ ([\d.]+) us$", log, re.M)]
            expected = (args.passes - args.warmup_passes) * len(case["frames"])
            if len(times) != expected:
                raise RuntimeError(f"{name}: expected {expected} timings, got {len(times)}")
            row = {"trial": trial, "model": name, "median_us": statistics.median(times),
                   "timed_evaluations": len(times), "stages_us": {k: float(v) for k, v in re.findall(
                       r"(exposure|input|network|expansion|reconstruction|downsample)\s+median ([\d.]+) us", log)}}
            if trial == 0:
                row["quality"] = [{"frame": i, **quality(f"{prefix}-{i}.rgba16f", resolve(frame["reference"]),
                                          *case["output"], case.get("exclude_rects", []))}
                                  for i, frame in enumerate(case["frames"]) if "reference" in frame]
            else:
                for i in range(len(case["frames"])):
                    Path(f"{prefix}-{i}.rgba16f").unlink()
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.out / "trials.json").write_text(json.dumps(rows, indent=1) + "\n")
    summary = {name: {"median_us": statistics.median(r["median_us"] for r in rows if r["model"] == name),
                      "stages_us": {k: statistics.median(r["stages_us"][k] for r in rows if r["model"] == name)
                                    for k in next(r for r in rows if r["model"] == name)["stages_us"]},
                      "quality": next(r["quality"] for r in rows if r["model"] == name and r["trial"] == 0)}
               for name, _ in models}
    for result in summary.values():
        q = result["quality"]
        if q and all(row["nonfinite_rgb"] == 0 for row in q):
            mse = statistics.mean(row["mse"] for row in q)
            result["pooled_mse"] = mse
            result["pooled_psnr_unit_peak_db"] = -10 * math.log10(mse) if mse else None
        if args.min_psnr is not None:
            if not q:
                raise ValueError("a PSNR gate requires at least one scored reference frame")
            limit = 10 ** (-args.min_psnr / 10)
            result["minimum_required_psnr_db"] = args.min_psnr
            result["meets_minimum_psnr"] = all(row["nonfinite_rgb"] == 0 and row["mse"] <= limit for row in q)
    (args.out / "summary.json").write_text(json.dumps({"case": case, "models": dict(models),
                                                     "summary": summary, "trials": rows}, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=MODEL_DIR")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-passes", type=int, default=12)
    parser.add_argument("--passes", type=int, default=37)
    parser.add_argument("--min-psnr", type=float, help="report whether every scored frame meets this unit-peak RGB PSNR")
    run(parser.parse_args())
