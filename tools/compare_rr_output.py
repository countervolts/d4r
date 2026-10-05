#!/usr/bin/env python3
"""Compares two d3d12_dlss_harness Ray Reconstruction runs.

The harness writes the presented image as a raw RGBA16F (or RGBA8) buffer, one
.frameN file per frame with D4R_HARNESS_SAVE_FRAMES=1 and, with the alpha
scenario, a .alpha.raw of the alpha texture. This tool compares two such runs -
in practice the same inputs with the shim's guide path on the GPU
(D4R_SHIM_RR_VRAM_GUIDES=1) and on the host (=0) - and says what differs:

  * image layout: every file has to be exactly width * height * bytesPerTexel,
  * rendered content: every RGB component's mean/max absolute difference, PSNR,
    and nonfinite samples, over the final image and every saved frame,
  * frame matching: frame N of one run can be required to match frame N of the
    other; optional temporal thresholds are explicit caller expectations,
  * alpha: the same comparison of the alpha buffers, plus the sentinel outside
    the caller's output subrect, which the denoiser must not touch.

Exit status is 0 only when every requested check passes.

usage: compare_rr_output.py --a RUN_A --b RUN_B --width W --height H
                            [--frames N] [--alpha] [--alpha-base X Y]
                            [--min-psnr DB] [--max-mean DIFF] [--alpha-sentinel VALUE]
"""

import argparse
import math
import os
import struct
import sys


def load(path, width, height, bpp):
    expected = width * height * bpp
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} bytes, expected {expected} "
                         f"({width}x{height} at {bpp} bytes per texel)")
    return data


def rgb_plane(path, width, height, bpp):
    """Compare every RGB component: luma alone can hide a red/blue swap."""
    raw = load(path, width, height, bpp)
    if bpp == 4:
        return [raw[index * 4 + channel] / 255.0
                for index in range(width * height) for channel in range(3)]
    return [component for pixel in struct.iter_unpack("<eeee", raw) for component in pixel[:3]]


def compare_planes(first, second, pixels, mask=None):
    """Mean/max absolute component difference, PSNR and nonfinite sample count."""
    total = 0.0
    peak = 0.0
    squared = 0.0
    counted = 0
    nonfinite = 0
    for index in range(pixels):
        if mask is not None and not mask(index):
            continue
        left, right = first[index], second[index]
        if not (math.isfinite(left) and math.isfinite(right)):
            nonfinite += 1
            continue
        delta = abs(left - right)
        total += delta
        squared += delta * delta
        peak = max(peak, delta)
        counted += 1
    if counted == 0:
        return {"mean": float("nan"), "max": float("nan"), "psnr": float("nan"),
                "nonfinite": nonfinite, "pixels": 0}
    mean = total / counted
    mse = squared / counted
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    return {"mean": mean, "max": peak, "psnr": psnr, "nonfinite": nonfinite, "pixels": counted}


def report(name, stats, failures, limits, require_same):
    print(f"  {name}: mean |delta| {stats['mean']:.6f}  max {stats['max']:.6f}  "
          f"PSNR {stats['psnr']:.2f} dB  nonfinite {stats['nonfinite']}  "
          f"pixels {stats['pixels']}")
    if stats["pixels"] == 0:
        failures.append(f"{name}: nothing to compare")
        return
    if require_same:
        if stats["psnr"] < limits["min_psnr"] or stats["mean"] > limits["max_mean"]:
            failures.append(f"{name}: PSNR {stats['psnr']:.2f} dB below {limits['min_psnr']}, "
                            f"mean {stats['mean']:.6f} above {limits['max_mean']}")
        if stats["nonfinite"]:
            failures.append(f"{name}: {stats['nonfinite']} nonfinite samples")


def alpha_checks(path, width, height, base_x, base_y, out_width, out_height, failures, sentinel):
    """Sentinel outside the subrect, finite values inside, coverage rising with x."""
    values = struct.unpack(f"<{width * height}f", load(path, width, height, 4))
    outside = 0
    invalid = 0
    columns = [0.0] * out_width
    counts = [0] * out_width
    for y in range(height):
        for x in range(width):
            value = values[y * width + x]
            inside = (base_x <= x < base_x + out_width and base_y <= y < base_y + out_height)
            if not inside:
                if value != sentinel:
                    outside += 1
                continue
            if not math.isfinite(value) or value < -0.001 or value > 1.001:
                invalid += 1
                continue
            local = x - base_x
            if 0 <= local < out_width:
                columns[local] += value
                counts[local] += 1
    means = [(columns[i] / counts[i]) if counts[i] else 0.0 for i in range(out_width)]
    rising = sum(1 for i in range(1, out_width) if means[i] >= means[i - 1])
    trend = rising / max(1, out_width - 1)
    print(f"  alpha {path}: outside-subrect writes {outside}, invalid {invalid}, "
          f"rising-column fraction {trend:.3f}, first {means[0]:.4f} last {means[-1]:.4f}")
    if outside:
        failures.append(f"{path}: {outside} pixels outside the output subrect were overwritten")
    if invalid:
        failures.append(f"{path}: {invalid} alpha values outside [0, 1] or nonfinite")
    if out_width > 1 and means[-1] - means[0] < 0.1:
        failures.append(f"{path}: alpha output lost the input's left-to-right coverage ramp")
    return trend


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="first run's output raw")
    parser.add_argument("--b", required=True, help="second run's output raw")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--rgba8", action="store_true", help="inputs were 8-bit, not 16-bit float")
    parser.add_argument("--frames", type=int, default=0,
                        help="compare this many .frameN files of both runs")
    parser.add_argument("--alpha", action="store_true", help="also compare the .alpha.raw buffers")
    parser.add_argument("--alpha-width", type=int, default=0)
    parser.add_argument("--alpha-height", type=int, default=0)
    parser.add_argument("--alpha-sentinel", type=float, default=-2.0,
                        help="expected value outside the alpha subrect (0 for UNORM destinations)")
    parser.add_argument("--alpha-base", type=int, nargs=2, default=(0, 0),
                        metavar=("X", "Y"))
    parser.add_argument("--min-psnr", type=float, default=45.0)
    parser.add_argument("--max-mean", type=float, default=0.002)
    parser.add_argument("--same-frames", action="store_true",
                        help="every saved frame of one run must match the other's")
    parser.add_argument("--temporal-change", type=float, default=None,
                        help="required mean |delta| between consecutive saved frames")
    parser.add_argument("--temporal-still", type=float, default=None,
                        help="upper bound on |delta| between consecutive saved frames of a run "
                             "whose inputs did not change")
    args = parser.parse_args()

    bpp = 4 if args.rgba8 else 8
    limits = {"min_psnr": args.min_psnr, "max_mean": args.max_mean}
    failures = []

    print(f"Comparing {args.a} with {args.b} at {args.width}x{args.height} "
          f"({4 if args.rgba8 else 8} bytes per texel)")
    first = rgb_plane(args.a, args.width, args.height, bpp)
    second = rgb_plane(args.b, args.width, args.height, bpp)
    pixels = args.width * args.height * 3
    report("final frame", compare_planes(first, second, pixels), failures, limits, True)

    if args.frames:
        for frame in range(1, args.frames + 1):
            path_a = f"{args.a}.frame{frame}"
            path_b = f"{args.b}.frame{frame}"
            if not (os.path.exists(path_a) and os.path.exists(path_b)):
                failures.append(f"frame {frame}: missing {path_a if not os.path.exists(path_a) else path_b}")
                continue
            plane_a = rgb_plane(path_a, args.width, args.height, bpp)
            plane_b = rgb_plane(path_b, args.width, args.height, bpp)
            report(f"frame {frame}", compare_planes(plane_a, plane_b, pixels), failures, limits,
                   args.same_frames)

    for run, plane in (("a", first), ("b", second)):
        if args.temporal_change is None and args.temporal_still is None:
            break
        previous = None
        for frame in range(1, args.frames + 1):
            path = f"{getattr(args, run)}.frame{frame}"
            if not os.path.exists(path):
                continue
            current = rgb_plane(path, args.width, args.height, bpp)
            if previous is not None:
                stats = compare_planes(previous, current, pixels)
                limit = args.temporal_change if args.temporal_change is not None else args.temporal_still
                if args.temporal_change is not None and stats["mean"] < limit:
                    failures.append(f"run {run} frame {frame}: consecutive frames differ by only "
                                    f"{stats['mean']:.6f}, the per-frame change did not reach the image")
                if args.temporal_still is not None and stats["mean"] > limit:
                    failures.append(f"run {run} frame {frame}: consecutive frames of an unchanged "
                                    f"input differ by {stats['mean']:.6f}")
                print(f"  run {run} frame {frame - 1} -> {frame}: mean |delta| {stats['mean']:.6f}")
            previous = current

    if args.alpha:
        width = args.alpha_width or args.width + 8
        height = args.alpha_height or args.height + 4
        alpha_a_path, alpha_b_path = args.a + ".alpha.raw", args.b + ".alpha.raw"
        missing = [path for path in (alpha_a_path, alpha_b_path) if not os.path.exists(path)]
        if missing:
            # An alpha run that wrote nothing is a failed comparison, not a reason to stop here.
            failures.append("alpha buffers missing: " + ", ".join(missing))
        else:
            alpha_failures = []
            trend_a = alpha_checks(alpha_a_path, width, height, args.alpha_base[0],
                                   args.alpha_base[1], args.width, args.height, alpha_failures, args.alpha_sentinel)
            trend_b = alpha_checks(alpha_b_path, width, height, args.alpha_base[0],
                                   args.alpha_base[1], args.width, args.height, alpha_failures, args.alpha_sentinel)
            alpha_a = struct.unpack(f"<{width * height}f", load(alpha_a_path, width, height, 4))
            alpha_b = struct.unpack(f"<{width * height}f", load(alpha_b_path, width, height, 4))
            stats = compare_planes(alpha_a, alpha_b, len(alpha_a))
            report("alpha", stats, alpha_failures, limits, True)
            if trend_a < 0.5 or trend_b < 0.5:
                alpha_failures.append("alpha coverage does not follow the input's left-to-right ramp")
            failures += alpha_failures
        for frame in range(1, args.frames + 1):
            paths = (f"{args.a}.alpha{frame}", f"{args.b}.alpha{frame}")
            missing = [path for path in paths if not os.path.exists(path)]
            if missing:
                failures.append(f"alpha frame {frame}: missing " + ", ".join(missing))
                continue
            planes = [struct.unpack(f"<{width * height}f", load(path, width, height, 4))
                      for path in paths]
            report(f"alpha frame {frame}", compare_planes(*planes, width * height),
                   failures, limits, args.same_frames)

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())