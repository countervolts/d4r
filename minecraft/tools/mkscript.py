#!/usr/bin/env python3 -I
"""Turn an nvxrt trace (NVXRT_TRACE=1 NVXRT_TRACE_RAW=1, patched layer from patches/nvxrt-resource-trace.patch) of a DLSS-G
harness run into a replayable per-frame script for the native Lite engine.

usage: mkscript.py LOG OUTDIR [--cache ~/.cache/nvxrt-gfx11] [--weights pre_*.bin] [--skip N]
One real frame period = from its input-upload group to the next frame's. Frames alternate (image ping-pong), so two
consecutive steady-state frames are written as script variants 0 and 1. Nothing here contains NVIDIA data except weights.bin,
which is a slice of the DLL's own weight buffer: keep it out of the repository.
"""
import re, sys, os, struct, argparse, glob, collections

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("out")
ap.add_argument("--cache", default=os.path.expanduser("~/.cache/nvxrt-gfx11"))
ap.add_argument("--weights", default=None, help="pre_*.bin dump of the arena buffer (NVXRT_DUMP/NVXRT_CAPTURE)")
ap.add_argument("--skip", type=int, default=2, help="real frames to skip before the steady-state pair")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

images, views, buffers, bdas, samplers, handles = {}, {}, {}, {}, {}, {}
modules, funcs = {}, {}
events = []          # ordered ops
cur = None
hexre = re.compile(r"nvxrt:\s+raw ([0-9a-f]+)")
for l in open(a.log, errors="replace"):
    l = l.rstrip("\n")
    m = hexre.match(l)
    if m and cur is not None:
        cur["raw"] = bytes.fromhex(m.group(1)); cur = None; continue
    if not l.startswith("nvxrt:"): continue
    l = l[7:]
    m = re.match(r"RES image (\S+) fmt (\d+) (\d+)x(\d+)x(\d+) layers (\d+) mips (\d+) usage (0x[0-9a-f]+) type (\d+) flags (0x[0-9a-f]+) tiling (\d+) samples (\d+)", l)
    if m:
        k = m.group(1); v = m.groups()
        images[k] = dict(fmt=int(v[1]), w=int(v[2]), h=int(v[3]), d=int(v[4]), layers=int(v[5]), mips=int(v[6]), usage=int(v[7], 16), type=int(v[8]), flags=int(v[9], 16)); continue
    m = re.match(r"RES view (\S+) image (\S+) fmt (\d+) type (\d+) base (\d+)/(\d+) layers (\d+)/(\d+)", l)
    if m:
        v = m.groups(); views[v[0]] = dict(image=v[1], fmt=int(v[2]), type=int(v[3]), mip=int(v[4]), mips=int(v[5]), layer=int(v[6]), layers=int(v[7])); continue
    m = re.match(r"RES buffer (\S+) size (\d+)", l)
    if m: buffers[m.group(1)] = int(m.group(2)); continue
    m = re.match(r"RES bda (\S+) addr ([0-9a-f]+) size (\d+)", l)
    if m: bdas[int(m.group(2), 16)] = (m.group(1), int(m.group(3))); continue
    m = re.match(r"CreateSampler (\S+): mag (\d+) min (\d+) mip (\d+) addr (\d+)/(\d+)/(\d+) border (\d+) unnorm (\d+)", l)
    if m: samplers[m.group(1)] = [int(x) for x in m.groups()[1:]]; continue
    m = re.match(r"ImageViewHandle view (\S+) \S+ -> (0x[0-9a-f]+) type (\d+) smp (\S+)", l)
    if m: handles[int(m.group(2), 16)] = (m.group(1), int(m.group(3)), m.group(4)); continue
    m = re.match(r"CreateCuModule: .* -> ([0-9a-f]{16}) \[ (.*) \]", l)
    if m:
        for n in m.group(2).split(): modules[n] = m.group(1)
        continue
    m = re.match(r"launch #(\d+) (\w+)\[(\w+)\] grid (\d+)x(\d+)x(\d+) block (\d+)x(\d+)x(\d+) smem (\d+) params (\d+)", l)
    if m:
        v = m.groups(); cur = dict(op="launch", n=int(v[0]), name=v[1], h12=v[2], grid=[int(x) for x in v[3:6]], block=[int(x) for x in v[6:9]], smem=int(v[9]), psize=int(v[10]))
        events.append(cur); continue
    m = re.match(r"EV fill buf (\S+) off (\d+) size (\d+) data (\d+)", l)
    if m: events.append(dict(op="fill", buf=m.group(1), off=int(m.group(2)), size=int(m.group(3)), val=int(m.group(4)))); continue
    m = re.match(r"EV clear img (\S+) color (\w+) (\w+) (\w+) (\w+)", l)
    if m: events.append(dict(op="clear", img=m.group(1), color=[int(m.group(i), 16) for i in (2, 3, 4, 5)])); continue
    m = re.match(r"EV buf2img buf (\S+) img (\S+) boff (\d+) rowlen (\d+) imgh (\d+) .* ext (\d+)x(\d+)x(\d+)", l)
    if m: events.append(dict(op="in", buf=m.group(1), img=m.group(2), boff=int(m.group(3)), w=int(m.group(6)), h=int(m.group(7)))); continue
    m = re.match(r"EV (copybuf|update|img2buf|copyimg)", l)
    if m: events.append(dict(op=m.group(1))); continue

launches = [e for e in events if e["op"] == "launch"]
starts = [i for i, e in enumerate(events) if e["op"] == "launch" and e["name"] == "Kernel_DownsampleRGBAndEstimateUIFused"]
# frame k begins at the group of "in" events right before its Downsample launch
def frame_begin(i):
    j = i
    while j > 0 and events[j - 1]["op"] in ("in", "copybuf", "update") : j -= 1
    return j
begins = [frame_begin(i) for i in starts]
first_launch = next(i for i, e in enumerate(events) if e["op"] == "launch")
init_clears = [e for e in events[:first_launch] if e["op"] == "clear" and any(e["color"][:4])]
for e in init_clears: pass
pair = [a.skip, a.skip + 1]
assert len(begins) > a.skip + 2, "not enough frames in the log"
frames = []
for k in pair + [0]:
    seg = events[begins[k]:begins[k + 1]]
    frames.append(seg)
print("frames (pair + init):", [len(f) for f in frames], "launches:", [sum(1 for e in f if e['op'] == 'launch') for f in frames])

# ------------------------------------------------------------------ symbolic parameters
def resolve_ptr(v):
    # bda base <= v < base+size
    best = None
    for base, (b, sz) in bdas.items():
        if base <= v < base + sz: best = (b, v - base); break
    return best

used_views, used_images, used_buffers, used_funcs, used_samplers = set(), set(), set(), set(), set()
hmap = {}  # handle value -> (view, kind, sampler)
def sym_params(e):
    p = e["raw"]; patches = []
    for o in range(0, len(p) - 7, 8):
        v = struct.unpack_from("<Q", p, o)[0]
        if not v: continue
        if (v >> 32) == 0 and v in handles:
            vw, ty, sm = handles[v]
            patches.append(("H", o, v)); used_views.add(vw); hmap[v] = (vw, ty, sm)
            if sm != "(nil)" and sm != "0": used_samplers.add(sm)
            continue
        r = resolve_ptr(v)
        if r:
            patches.append(("P", o, r[0], r[1])); used_buffers.add(r[0])
    return patches

# kernel param layout: matrix patch for EstimatePrev2CurrScatter
lines = []
for vi, f in enumerate(frames):
    ops = []
    for e in f:
        if e["op"] == "launch":
            e["patches"] = sym_params(e); used_funcs.add((e["name"], e["h12"]))
            ops.append(e)
        elif e["op"] == "fill":
            used_buffers.add(e["buf"]); ops.append(e)
        elif e["op"] == "clear":
            used_images.add(e["img"]); ops.append(e)
        elif e["op"] == "in":
            used_images.add(e["img"]); ops.append(e)
    frames[vi] = ops

for v in list(used_views): used_images.add(views[v]["image"])
for e in init_clears: used_images.add(e["img"])
# function -> full hash (module dir prefix)
kd = os.path.join(a.cache, "kernels")
full = {}
for d in os.listdir(kd): full[d[:16]] = d
def modhash(name, h12):
    for d in os.listdir(kd):
        if d.startswith(h12): return d
    raise SystemExit("no module for %s %s" % (name, h12))

out = []
out.append("d4rfg 1")
for k in sorted(used_images):
    i = images[k]; out.append("image %s %d %d %d %d %d %d 0x%x 0x%x %d" % (k, i["fmt"], i["w"], i["h"], i["d"], i["layers"], i["mips"], i["usage"], i["flags"], i["type"]))
for k in sorted(used_views):
    v = views[k]; out.append("view %s %s %d %d %d %d %d %d" % (k, v["image"], v["fmt"], v["type"], v["mip"], v["mips"], v["layer"], v["layers"]))
for k in sorted(used_samplers):
    out.append("sampler %s %s" % (k, " ".join(str(x) for x in samplers[k])))
for k in sorted(used_buffers): out.append("buffer %s %d" % (k, buffers[k]))
for hv, (vw, ty, sm) in sorted(hmap.items()):
    out.append("handle 0x%x %s %d %s" % (hv, vw, ty, sm if sm not in ("(nil)", "0") else "-"))
fidx = {}
for (n, h) in sorted(used_funcs):
    fidx[(n, h)] = len(fidx); out.append("func %d %s %s" % (fidx[(n, h)], modhash(n, h), n))
# initial data
arena = None
if a.weights:
    sz = os.path.getsize(a.weights)
    cands = [k for k in used_buffers if buffers[k] == sz]
    assert len(cands) == 1, cands
    arena = cands[0]
    import shutil; shutil.copyfile(a.weights, os.path.join(a.out, "weights.bin"))
    out.append("initdata %s weights.bin" % arena)
for e in init_clears:
    out.append("initclear %s %d %d %d %d" % (e["img"], *e["color"]))
for vi, ops in enumerate(frames):
    out.append("frame %d" % vi); seen_lab = set(); gen = 0
    for e in ops:
        if e["op"] == "launch":
            ps = ",".join("%s:%d:%s" % (p[0], p[1], ":".join(str(x) for x in p[2:])) if p[0] == "P" else "H:%d:0x%x" % (p[1], p[2]) for p in e["patches"]) or "-"
            out.append("launch %d %d %d %d %d %d %d %d %d %s %s" % (fidx[(e["name"], e["h12"])], *e["grid"], *e["block"], e["smem"], e["psize"], e["raw"].hex(), ps))
            if e["name"] == "Kernel_Copy4Channel":
                hv = struct.unpack_from("<Q", e["raw"], 16)[0]
                out.append("output %d %s" % (gen, views[handles[hv][0]]["image"])); gen += 1
            if e["name"] == "Kernel_BlendCandidatesFGLite":
                hv = struct.unpack_from("<Q", e["raw"], 128)[0]
                out.append("output %d %s" % (gen, views[handles[hv][0]]["image"])); gen += 1
                used_images.add(views[handles[hv][0]]["image"])
        elif e["op"] == "fill":
            out.append("fill %s %d %d %d" % (e["buf"], e["off"], e["size"], e["val"]))
        elif e["op"] == "clear":
            out.append("clear %s %d %d %d %d" % (e["img"], *e["color"]))
        elif e["op"] == "in":
            fm = images[e["img"]]["fmt"]
            if fm == 37:
                lab = "backbuffer" if "backbuffer" not in seen_lab else "hudless"
            else:
                lab = "depth" if fm == 100 else "mvec"
            seen_lab.add(lab)
            out.append("input %s %s %d %d" % (lab, e["img"], e["w"], e["h"]))
    out.append("endframe")
open(os.path.join(a.out, "script.txt"), "w").write("\n".join(out) + "\n")
print("wrote", len(out), "lines;", len(used_images), "images", len(used_views), "views", len(used_buffers), "buffers", len(fidx), "funcs")
