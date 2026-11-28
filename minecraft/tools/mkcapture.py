#!/usr/bin/env python3 -I
"""Synthetic Minecraft-convention capture for the DLSS-G harness replay mode (fg/dlssg/harness --replay DIR) and for fg_replay_test.
Native-resolution inputs: RGBA8 colour, D32 reversed-Z depth, RG16F motion in pixels (previous = current + mv), scale 1/size.
usage: mkcapture.py OUTDIR [--w 2560 --h 1440 --frames 8 --template ~/.cache/dlssg-capture-got --truth DIR]"""
import argparse, os, numpy as np, re
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--w", type=int, default=2560); ap.add_argument("--h", type=int, default=1440)
ap.add_argument("--frames", type=int, default=8); ap.add_argument("--mfc", type=int, default=3)
ap.add_argument("--template", default=os.path.expanduser("~/.cache/dlssg-capture-got")); ap.add_argument("--truth", default=None)
a = ap.parse_args(); W, H = a.w, a.h
os.makedirs(a.out, exist_ok=True)
pan, sqvx, sqvy, sq = 6.0, 23.0, 11.0, 160
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); px, py = xx + 0.5, yy + 0.5
def render(t):
    sx, sy = 400 + sqvx * t, 300 + sqvy * t
    bx = px + pan * t
    cx, cy = np.floor(bx / 64).astype(int), np.floor(py / 64).astype(int)
    v = np.where(((cx + cy) & 1) == 1, 200, 60).astype(np.float32)
    img = np.stack([v, v * 0.9, v * 0.7, np.full_like(v, 255)], -1)
    wl = (np.floor(bx).astype(int) & 15) == 0
    img[wl, 0:3] = 255
    ins = (px >= sx) & (px < sx + sq) & (py >= sy) & (py < sy + sq)
    sqc = np.stack([np.full_like(px, 230), 40 + (px - sx) / sq * 180, 40 + (py - sy) / sq * 180, np.full_like(px, 255)], -1)
    img[ins] = sqc[ins]
    return np.clip(img, 0, 255).astype(np.uint8), ins
def aux(t):
    _, ins = render(t)
    depth = np.where(ins, 0.9, 0.1).astype(np.float32)           # reversed-Z: near = large
    mv = np.zeros((H, W, 2), np.float32); mv[..., 0] = np.where(ins, -sqvx, pan); mv[..., 1] = np.where(ins, -sqvy, 0.0)
    return depth, mv.astype(np.float16)
for r in range(a.frames):
    img, _ = render(float(r)); d, mv = aux(float(r))
    for name, data, fmt, pitch in (("Backbuffer", img, 28, W * 4), ("HUDLess", img, 28, W * 4), ("Depth", d, 20, W * 4), ("MVecs", mv, 34, W * 4)):
        open(os.path.join(a.out, f"res_r{r:02d}_{name}_{W}x{H}_fmt{fmt}_pitch{pitch}.bin"), "wb").write(data.tobytes())
    for e in range(1, a.mfc + 1):
        src = open(os.path.join(a.template, f"params_r{min(r, 7):02d}_e{min(e, 3)}.txt")).read().split("\n")
        out = []
        for l in src:
            if re.match(r"[fu] DLSSG\.(Width|Height)\b", l): l = l.split()[0] + " " + l.split()[1] + " " + str(W if "Width" in l else H)
            elif re.match(r"u DLSSG\.(MVecs|Depth|HUDLess|Backbuffer)Subrect(Width|Height)", l): l = l.split()[0] + " " + l.split()[1] + " " + str(W if "Width" in l else H)
            elif l.startswith("f DLSSG.MvecScaleX"): l = "f DLSSG.MvecScaleX %.9g" % (1.0 / W)
            elif l.startswith("f DLSSG.MvecScaleY"): l = "f DLSSG.MvecScaleY %.9g" % (1.0 / H)
            elif l.startswith("f DLSSG.JitterOffset"): l = l.split()[0] + " " + l.split()[1] + " 0"
            elif l.startswith("u DLSSG.DepthInverted"): l = "u DLSSG.DepthInverted 1"
            elif l.startswith("u DLSSG.MvecJittered"): l = "u DLSSG.MvecJittered 0"
            elif l.startswith("u DLSSG.MultiFrameIndex"): l = "u DLSSG.MultiFrameIndex %d" % e
            elif l.startswith("u DLSSG.MultiFrameCount"): l = "u DLSSG.MultiFrameCount %d" % a.mfc
            elif l.startswith("m DLSSG.ClipToPrevClip"): l = "m DLSSG.ClipToPrevClip 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
            elif l.startswith("r DLSSG.") and re.search(r"\d+x\d+", l):
                l = re.sub(r"\d+x\d+", f"{W}x{H}", l, count=1)
            out.append(l)
        open(os.path.join(a.out, f"params_r{r:02d}_e{e}.txt"), "w").write("\n".join(out))
if a.truth:
    os.makedirs(a.truth, exist_ok=True)
    for r in range(a.frames):
        for k in range(1, a.mfc + 1):
            for tag, t in (("p", r - 1 + k / (a.mfc + 1)), ("n", r + k / (a.mfc + 1))):
                img, _ = render(t)
                with open(os.path.join(a.truth, f"r{r:02d}_g{k}_{tag}.ppm"), "wb") as f:
                    f.write(b"P6\n%d %d\n255\n" % (W, H)); f.write(img[..., :3].tobytes())
print("ok")
