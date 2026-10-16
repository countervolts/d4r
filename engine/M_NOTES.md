# Preset M (rrlite) - reverse-engineering notes

Working notes for the engine's own preset M stages. Sources: captured launches (`build/engine-work/mq`), the
translated kernels (`vulkan/ptx2glsl.py` + `vulkan/condense.py` on the DLL's PTX; M uses the `_folded` entry
points: `rrlite_enc0_4x4_mvlo_hdr_folded`, `rrlite_dec0_4x4_folded`, `rrlite_post_3_1_mvlo_hdr_folded`), and
`kernels/tools/swin_model.py` for the network.

## Frame (640x360 render -> 1280x720 output in the capture)
1. exposure: `cuda_dldn_engine_luma_convert_kernel`, 2x `reduce_sum`, `auto_exposure_copy` (every frame for M)
2. `rrlite_enc0_*_folded` grid 81x49 blocks of 32 lanes: features from textures -> 16 tokens x 128 FP8 features per
   block -> GEMM 128 -> 64 (+ bias, clamp +-2pi) -> tokens 320x192 x 64 ch (FP8, planes of 32) ; also writes three
   640x360 2-channel f16 surfaces (args 280/288/296)
3. enc1 (C 64, 2 heads, shift 2, merge -> 96), enc2 (C 96, 4 heads, shift 2, merge -> 128), 6 tube blocks (C 128,
   4 heads, 80x48 tokens, shifts 2,0,2,0,2,0, own weights each), dec2 (C 96, expand from 128, shift 0),
   dec1 (C 64, expand from 96, shift 0). Fixed shifts (no per-frame schedule seen).
4. `rrlite_dec0_4x4_folded` grid 80x48 blocks of 32 lanes, P16 = (640, 384) = the token grid x2: per block the 16
   low-resolution tokens (mirrored at the borders) x 64 ch -> GEMM to 4 sub-positions x 32 values (weights at
   w + 16 lane + 512 i, f16 bias pairs at w + 8192) -> per render pixel outputs on four 640x360 surfaces:
   128 SNORM8x4, 136 UNORM8x4, 144 UNORM8x4, 152 RGBA16F. Textures 56 (2ch f16, point) and 64 (SNORM8x4, linear).
5. `rrlite_post_3_1_mvlo_hdr_folded` grid 120x34 blocks of 16x16 threads, two output rows per thread: the
   reconstruction at 3x render resolution (1920x1080). Textures: 0 colour (linear), 16/24 exposure (1x1),
   32/40 UNORM8x4 from dec0 (point), 48 RGBA16F from dec0 (point), 56 previous 1920x1080 result (linear),
   64/72 2ch f16 (point), 80 2ch f16 (linear). Writes surface 96 (1920x1080 RGBA16F, next history), surface 112,
   and a packed 11-11-10 copy to the buffer at 104. Shared tiles (16 rows x 10 x 8 bytes): colour + mvec.x,
   mvec.yz + ajab.xy, ajab.z + aniso.
6. `rrlite_downsample_kernel_static_hdr`: packed 1920x1080 -> 1280x720 output. Own version: `down_m.comp`.

NGX texture filters (cuTexObjectCreate): enc0: 80 linear, 96 point, 112/120 point (exposure), 128 linear,
136 point, 152 linear, 160 linear. dec0: 56 point, 64 linear. post: see above. downsample: 48 point, 56 linear.

Array formats in the dumps: fmt194 = UNORM8x4, fmt200 = SNORM8x4, fmt16 = f16, fmt32 = f32.

## dec0 (`rrlite_dec0_4x4_folded`), verified numerically against a capture (`build/engine-work/mre/dec0_check.py`)
Weights: FP8 at w, the Swin fragment layout `woff_table(0, 512, 1024, K 64, N 128)`; bias 128 f16 at w + 8192,
plain order. E = tokens(dec1 output, 64 ch) W + bias, per low-resolution token; render pixel (x, y) takes columns
32 q + j of token (x/2, y/2), q = 2 (y & 1) + (x & 1), each rounded to FP8: v[j]. Only j = 0..16 are used.
- S128 (SNORM8x4, the recurrent feature): out_i = (1 - g_i) hist_i + g_i tanh(v[2 + i]),
  g_i = .5 tanh(.5 v[6 + i]) + .5, i = 0..3, f16 arithmetic. hist = previous S128 (tex 64, linear) at the
  reprojected position, 0 when reset (P120 byte) or off screen:
  f = (pixel + .5 - P72) * P88 + mv (tex 56, texelFetch at the pixel; P88 = 3: units of the 3x image),
  off screen if f < 0 or f > P96 (1920, 1080); uv = P104 * ((f / P96 + P80 / P112) * P112), P112 = (640, 360),
  P104 = 1 / P112.
- S136 (UNORM8x4): sigmoid(v0), sigmoid(v1), sigmoid(v10), 0 (constant in the capture).
- S144 (UNORM8x4): softmax(v11, v12, v13), alpha 0.
- S152 (RGBA16F): a = P160 f16(sigmoid(clamp(v14, +-20))), b likewise with v15, t = v16 * P164,
  rho = tanh(t) (|t| >= 9.01: 1; >= .6: 1 - 2 / (exp2(2.88539 |t|) + 1) with the sign of t; else t + t * poly(t^2),
  poly = ((.0157397 s - .052304) s + .133153) s - .3333277) s); out = (a^2, b^2, a b rho, 0). P160 = 8, P164 = .0786.

## post (`rrlite_post_3_1_mvlo_hdr_folded`) - in progress
Block 16x16 threads; thread (tx, ty) handles hi-res pixels (x, 2y) and (x, 2y + 1) (pairs as f16x2). Tile of
render texels, 10 columns x 16 rows, origin (floor(P124 * 16 bx) - 2, floor(P128 * 32 by) - 2), clamped to
[0, P148 - 1] x [0, P152 - 1] (+ colour origin P8/P12), three 8-byte-per-texel arrays:
 tile0: range-compressed colour c' = log10(max(100 * min(e1 * c, 32752) + 1, 1e-4)) * 0.36 (rgb, f16; e1 = tex16, 0 -> 1), T40.x
 tile1: T40.y, T40.z, T32.x, T32.y        (T32, T40: the UNORM8x4 outputs of dec0)
 tile2: T32.z, -0.721325 * cov.x, -0.721325 * cov.y, -1.44265 * cov.z      (cov = tex48 = dec0's covariance)
Render position of a pixel centre: f4 = (x + .5) * P124 + P132 (P124 = 1/3), f9/f11 likewise in y with P128, P136.
Main loop: the 3x3 render texels around floor(f - .5); per texel an anisotropic Gaussian
w = exp2(A dx^2 + B dy^2 + C dx dy) with (A, B, C) that texel's tile2 values and (dx, dy) = texel centre - position
(f16); every tile quantity is accumulated with w (kernel-predicted resampling).
After the loop (per pixel, f32): cur = sum(w * colour') / sum(w) (rgb, compressed), m = resampled T40.xyz,
g = resampled T32.xyz.
Motion: three candidates, mvA = texelFetch(tex64) and mvB = texelFetch(tex72) at the nearest render texel
(floor(f4), floor(f9)), mvC = tex80 (linear, uv = position * P160/P164); paired with the three weights m; the two
with the largest weights are kept (w1 >= w2). k = clamp(1.1 w - .05, 0, 1); b2 = k2 / max(k1 + k2, 1e-10), set to 0
when below 1/255 (then b1 = 1), b1 = k1 / (k1 + k2).
History sample for a candidate: q = pixel + .5 + mv (hi-res pixels). Off screen (q.x < 0, q.y < 0, q.x > P140,
...): the current colour texel instead (nearest render texel, times e2 = tex24, compressed as above). Otherwise
tex56 (previous result) with the 5-tap bilinear Catmull-Rom cross: base = floor(q - .5) + .5, t = sat(q - base),
w0 = t^2 - (t + t^3)/2, w1 = 1.5 t^3 - 2.5 t^2 + 1, w3 = (t^3 - t^2)/2, w2 = 1 - w0 - w1 - w3, w12 = w1 + w2,
taps at (base.x - 1, base.y + w2y/w12y), (base.x + w2x/w12x, base.y - 1), centre, (.., base.y + 2), (base.x + 2, ..)
with weights w0x w12y, w12x w0y, w12x w12y, w12x w3y, w3x w12y, normalised by their sum. uv = q' / (P140, P144).
Details of the loop: base texel r6 = floor(f4 - .5), r8 = floor(f9 - .5); taps are texels r6 + i, r8 + j, i, j in
-1..1, dx = (r6 + .5 - f4) + i, dy likewise (f16). e = (dx^2 A + dy^2 B) + (dx dy) C in f16, w = exp2(e): the
products use f16(w), the weight sum W the unrounded f32 values. Accumulators are f16 fma chains (i outer, j inner).
tile0 -> cur'.rgb, m.x ; tile1 -> m.y, m.z, g.x, g.y ; tile2 -> g.z (then A, B, C). All divided by W.
T40 = dec0's mix (softmax) -> m ; T32 = dec0's gates -> g.
Candidate order: a = (mvA, m.x), c = (mvC, m.y), b = (mvB, m.z). big, small = (a.w < c.w) ? (c, a) : (a, c);
s = !(small.w < b.w) ? small : b; if (big.w < s.w) swap(big, s). Candidate 1 = big, candidate 2 = s.
Candidate 2's history is 0 (not the fallback) when q.y > P144 or the reset byte (P168) is set.
Blend (f32): hist' = b1 h1 + b2 h2 ; expand(v) = (10^(clamp(v, 1e-4, 3) / .36) - 1) / 100 ;
histLin = e1 * expand(hist') / e2 ; curLin = expand(cur') ; a0 = clamp(P120 * W * g.x, 0, 1) ;
a1 = a0 (1 - g.y) + g.y ; a2 = a0 (1 - g.z) + g.z ; outA = hist + (cur - hist) a1 ; outB = hist + (cur - hist) a2 ;
a non-finite out is replaced by cur. compress(v) = log10(max(100 v + 1, 1e-4)) * .36.
Outputs: surface 96 (x, y) = (compress(outA), 0) - the next frame's history ;
buffer 104 [y * P140 + x] = 11-11-10 bits of compress(outB) * .4081632 (trunc(v * 2047 + .5) clamped; blue 1023) ;
surface 112 (x / 2, y / 2) = compress(mean of outA over the 2x2 pixels) * .4081632.
e1 = tex16 (current exposure), e2 = tex24 (previous), 0 -> 1.

## enc0 (`rrlite_enc0_4x4_mvlo_hdr_folded`) - in progress
Block = 8x8 render pixels (32 lanes x 2 pixels) = 4x4 tokens of 2x2 pixels; pixel = 8 * block - P8 (2, 2) + (lane & 7,
lane >> 3) (+ 4 rows for the lane's second pixel), mirrored into the image (|v|, and 2 n - v - 2 beyond P56 = render
size); mirrored pixels are flagged (no surface stores).
Per pixel: exposures tex112 (e1) / tex120 ; colour tex80 texel (origin P88, max P232) ; depth tex136 over the 3x3
neighbourhood (origin P144), keeping the positions of the smallest and largest depth (the byte at P216 swaps their
roles) ; motion tex96 at those two positions and at the pixel -> the three candidates written to surfaces
280 / 288 / 296 (two f16 each; post's mvA / mvB / mvC).
Motion: mv = P200 * (P208 + tex96 texel) (P200 = 3: hi-res pixels). A = at the 3x3 neighbour with the smallest depth
when the P216 byte is set (largest otherwise), B = at the other extreme, C = at the pixel. Strict comparisons, the
centre first, then (-1,-1) (0,-1) (1,-1) (-1,0) (1,0) (-1,1) (0,1) (1,1).
32 feature slots per pixel, FP8, two planes of 16 bytes:
 plane 0: R' G' | f.x f.y | B' h1.R | f.z f.w | h1.G h1.B | 0 0 | h2.R h2.G | 0 0
 plane 1: h2.B h3.R | 0 0 | h3.G h3.B | 0 0 | h4.R h4.G | 0 0 | h4.B mf | 0 0
 (R', G', B') = compress(e1 * colour) (with the 32752 clamp and non-finite -> 1 before the log, as in post)
 h1..h4 = compress(e1 * s_k), s_k = history colour along a candidate: q = mv + (pixel + .5 - P168) * P192 for
   k = 1 (A), 2 (C), 3 (B); k = 4: C again with q = mv + (pixel + .5) * P192 (no jitter term).
   off screen (q < 0, q > P248) or reset byte P316: s = the pixel's own colour texel (linear);
   otherwise tex152 (the half-resolution output of the previous post, linear filter, uv = q * P256) * 2.45, and when
   e2 = tex120 > 0: expand(.) / e2.
 f = tex160 (dec0's recurrent feature, SNORM8, linear) at uv = (q_A / P248 + P176 / P56) ; 0 when q_A is off
   screen or reset.
 mf = P272 * ln(|d| + 1) (0 when reset), d = P268 * 540 / P60 * P184 * (mvA - prevA), prevA = tex128 (the previous
   frame's surface 280, linear) at uv = clamp((pixel + .5 - P168 + P176 + mvA * P184) * P64, 0, P56 * P64).
 Each value: non-finite -> 0, min(v, 1024), then dithered truncation to FP8: n = u16 at w[2 * i] >> 9 with
 i = (P312 << 8) | (|y| & 15) << 4 | (|x| & 15) (to be confirmed), v' = bits((v + f16(bits(v) & 0xFC00 | n)) -
 f16(bits(v) & 0xFC00)) & 0xFF80 in f16 arithmetic.
Token = its 2x2 pixels: K index 32 q + 16 plane + byte, q = 2 (py & 1) + (px & 1). Weights at w + 4096:
`woff_table(4096, 512, 2048, K 128, N 64)`, bias 64 f16 at w + 12288; output clamp(E, +-2 pi) stored as FP8.
The N columns (and the bias) are in pair order: natural channel c is column gp(c) (`swin_model.gp`). Verified by
regression on the reset frame (`build/engine-work/mre/enc0_check.py`): slots 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28.
The first 4096 bytes of the blob are the dither table (8 x 256 u16).

## exposure (cuda_dldn_engine_* kernels), every frame
sum = mean over render pixels of min(max(.25 R + .5 G + .25 B, 0), 10000) ; exposure = 0.23547077 / max(sum, 0.005)
(f64 division, stored as f16). Capture: mean 0.3455 -> 0.6816. The 0.2195 in the parameter block is not used.

## Sizes and parameters across modes (harness, 2026-10-08)
- Token pixel grid P16 = render size rounded up to multiples of 32; tokens = / 2; the tube works at / 8.
- The reconstruction image is 1.5 x the OUTPUT (2560x1440 -> 3840x2160, 3840x2160 -> 5760x3240), not 3 x render; the
  downsample is always the static 3 -> 2 one. P124 (post) = render / reconstruction size.
- `rrlite_post_3_1` runs when reconstruction / render >= 3 (2x and 3x upscaling); `rrlite_post_3_2` when it is smaller
  (1.5x: 2.25, balanced: 2.59). post_m.comp is post_3_1 only: the engine covers Performance and Ultra Performance.
- Constants in every mode: post gain 2.75301 ; enc0 change scale 9.04507 and gain 0.22357 ; dec0 sigma 8, rho 0.07863.
- jitter = NGX's Jitter.Offset as given ; enc0 P176 / dec0 P80 = the previous frame's jitter ; enc0/dec0 motion scale
  and render->hi factors = reconstruction / render ; downsample P80 = render / output, P88 = jitter.
