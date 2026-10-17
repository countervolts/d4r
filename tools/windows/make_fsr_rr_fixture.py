"""Synthetic noisy DLSS-RR-style scene with a known noise-free answer, in the rr_ngx_probe fixture format.

Left-handed view space, linear depth, render size == output size or output larger. Surfaces: a wall (stripes fixed in
WORLD space), a disc (a sphere's normals) and a sky band. The colour is (diffuse + specular albedo) * smooth lighting
times per-frame mean-preserving log-normal noise; the sky is noise-free. The camera may translate along world x
(--camera-speed per frame): the guides then carry the exact motion vectors (previous minus current, render pixels) and
the view matrix, and clean-<frame>.bin is the noise-free output-size image of that frame. clean.bin is frame 0's.
--extents W1xH1,W2xH2,... makes the render size change per frame (cycling, each within --size): the guides are written
at that frame's size and dynamic-extents.txt lists them (the probe's "dynamic-extents" mode).
No game, no NVIDIA runtime. Output stays under ignored build/.
"""
import argparse
import math
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / '.tools/python/vendor'))
import numpy as np

FOV_DEGREES = 60.0


def scene(width, height, cam_x=0.0, prev_cam_x=None):
    """Returns depth, normal, diffuse, specular, clean colour, sky mask and motion (render pixels, prev - current)."""
    if prev_cam_x is None:
        prev_cam_x = cam_x
    y, x = np.mgrid[0:height, 0:width].astype('f8')
    u, v = (x + 0.5) / width * 2 - 1, 1 - (y + 0.5) / height * 2
    aspect = width / height
    tangent = math.tan(math.radians(FOV_DEGREES) / 2)
    # camera-space ray scale: x_view = u * aspect * tangent * z, y_view = v * tangent * z
    z = np.full((height, width), 10.0)
    normal = np.zeros((height, width, 3)); normal[..., 2] = -1.0
    diffuse = np.empty((height, width, 3)); diffuse[:] = (0.60, 0.50, 0.40)
    # the disc: a world-fixed sphere of radius 0.9 centred at world (0.9, -0.5, 7.5)
    centre = np.array([0.9, -0.5, 7.5]); radius = 0.9
    ray = np.stack([u * aspect * tangent, v * tangent, np.ones_like(u)], axis=-1)
    origin = np.array([cam_x, 0.0, 0.0])
    oc = origin - centre
    b = np.einsum('ijk,k->ij', ray, oc); a = np.einsum('ijk,ijk->ij', ray, ray); c = float(oc @ oc - radius * radius)
    disc_term = b * b - a * c
    hit = disc_term > 0
    t = np.where(hit, (-b - np.sqrt(np.maximum(disc_term, 0))) / a, 0)
    point = origin + ray * t[..., None]
    disc = hit & (t > 0)
    z[disc] = (t * ray[..., 2])[disc]  # view z = t (the ray's z component is 1)
    normal[disc] = ((point - centre) / radius)[disc]
    world_x = cam_x + u * aspect * tangent * 10.0
    stripe = (np.floor(world_x / 0.45).astype(int) % 2) == 0
    diffuse[stripe & ~disc] = (0.35, 0.45, 0.55)
    diffuse[disc] = (0.80, 0.25, 0.20)
    sky = v > 0.7
    z[sky] = 1.0e8; normal[sky] = 0.0; diffuse[sky] = 0.0; disc = disc & ~sky
    specular = np.full((height, width, 3), 0.04); specular[sky] = 0.0
    # lighting is a function of the surface point (world x, screen-independent y), so it moves with the scene
    surf_x = np.where(disc, point[..., 0], world_x)
    light = (1.5 + 0.8 * np.sin(surf_x * 2.2) * np.cos(y * 360.0 / height / 41.0))[..., None] * np.array([1.0, 0.92, 0.8])
    clean = (diffuse + specular) * light
    clean[sky] = (0.35, 0.55, 0.95)
    # motion: previous minus current screen position of the same surface point, in render pixels
    motion = np.zeros((height, width, 2))
    dx = cam_x - prev_cam_x
    if dx != 0.0:
        depth = np.where(sky, 1.0, z)
        motion[..., 0] = dx / (depth * tangent * aspect) * (width / 2.0)
        motion[sky] = 0.0
    return z, normal, diffuse, specular, clean, sky, motion


def half_bytes(a):
    return np.ascontiguousarray(a.astype('<f2')).tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--size', default='640x360', help='render size (the feature creation size)')
    parser.add_argument('--out-size', help='output size (default: render size)')
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--noise', type=float, default=0.6, help='log-normal sigma of the per-frame colour noise')
    parser.add_argument('--camera-speed', type=float, default=0.0, help='camera translation along world x per frame')
    parser.add_argument('--extents', help='per-frame render sizes WxH,WxH,... cycling (each within --size, aspect ratio kept)')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((repo / 'build').resolve()):
        parser.error('output must stay under ignored build/')
    args.output.mkdir(parents=True, exist_ok=False)
    width, height = map(int, args.size.split('x'))
    out_width, out_height = map(int, (args.out_size or args.size).split('x'))
    extents = [tuple(map(int, item.split('x'))) for item in args.extents.split(',')] if args.extents else [(width, height)]
    for ew, eh in extents:
        if ew < 64 or eh < 64 or ew > width or eh > height:
            parser.error('every extent must be at least 64 and within --size')
    rng = np.random.default_rng(args.seed)
    f = 1.0 / math.tan(math.radians(FOV_DEGREES) / 2)
    near, far = 0.1, 1000.0
    lines = [f'size {width} {height} {out_width} {out_height}']
    extent_lines = []
    for i in range(args.frames):
        fw, fh = extents[i % len(extents)]
        extent_lines.append(f'{fw} {fh}')
        aspect = fw / fh
        proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (far - near), 1, 0, 0, -near * far / (far - near), 0]
        cam = args.camera_speed * i
        z, normal, diffuse, specular, clean_render, sky, motion = scene(fw, fh, cam, cam - args.camera_speed if i else cam)
        if (out_width, out_height) == (fw, fh):
            clean_out, sky_out = clean_render, sky
        else:
            clean_out, sky_out = scene(out_width, out_height, cam)[4:6]
        noise = np.exp(args.noise * rng.standard_normal((fh, fw, 3)) - 0.5 * args.noise ** 2)
        noise[sky] = 1.0
        color = np.concatenate([clean_render * noise, np.ones((fh, fw, 1))], axis=-1)
        rgba = lambda c3: np.concatenate([c3, np.zeros((fh, fw, 1))], axis=-1)
        (args.output / f'{i}-color.bin').write_bytes(half_bytes(color))
        (args.output / f'{i}-motion.bin').write_bytes(half_bytes(motion))
        (args.output / f'{i}-depth.bin').write_bytes(np.ascontiguousarray(z.astype('<f4')).tobytes())
        (args.output / f'{i}-normals.bin').write_bytes(half_bytes(rgba(normal)))
        rough = np.zeros((fh, fw, 4)); rough[..., 0] = 0.5
        (args.output / f'{i}-roughness.bin').write_bytes(half_bytes(rough))
        alb = lambda a: np.concatenate([np.round(a * 255), np.full((fh, fw, 1), 255)], axis=-1).astype('u1').tobytes()
        (args.output / f'{i}-diffuse.bin').write_bytes(alb(diffuse))
        (args.output / f'{i}-specular.bin').write_bytes(alb(specular))
        view = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -cam, 0, 0, 1]  # row-vector world -> view: translate by -camera
        lines.append('frame %d 0 0 1 1 1 %d 0 ' % (i, 1 if i == 0 else 0) + ' '.join(f'{v:.9g}' for v in view + proj))
        (args.output / f'clean-{i}.bin').write_bytes(half_bytes(np.concatenate([clean_out, np.ones((out_height, out_width, 1))], axis=-1)))
        if (out_width, out_height) != (fw, fh):  # the best a render-resolution result could be, before upscaling
            (args.output / f'clean-render-{i}.bin').write_bytes(half_bytes(np.concatenate([clean_render, np.ones((fh, fw, 1))], axis=-1)))
        if i == 0:
            (args.output / 'clean.bin').write_bytes((args.output / 'clean-0.bin').read_bytes())
            np.save(args.output / 'sky.npy', sky_out)
    (args.output / 'frames.txt').write_text('\n'.join(lines) + '\n')
    if args.extents:
        (args.output / 'dynamic-extents.txt').write_text('\n'.join(extent_lines) + '\n')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
