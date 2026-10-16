# K output kernel (hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel) - reverse-engineering notes
P = parameter block (328 bytes). rd1 = P+256, rd17 = P. Offsets below are absolute in P.
Textures: 200 color (linear), 224 history color (linear), 232 hi-res history 2x (linear), 240 motion (point), 248 depth (point),
          256 exposure prev? (1x1), 264 exposure cur? (1x1). Buffers: 272 (network head, 40 ch?), 280. Surfaces: 288 out/history color,
          296 hi-res history (R16, byte stores), 304 320x256 x4, 312 final output.
## entry
 eA = tex264(.5,.5).r (0 -> 1), eB = tex256(.5,.5).r (0 -> 1)
 f497 = P160.x*eA ; f503 = P160.x*eB ; f1 = P168*f503 (0 -> 1) ; f2 = f503 / f497
 block origin (r1,r2) = 16*ctaid. render-space tile: u = (px+0.5)*P40 + P16 ; r3,r4 = roundHalfAway(u(first)) - 4 ; r5,r6 = roundHalfAway(u(last)) + 4
 r7 = r5-r3+1 (tile width) ; r8 = r6-r4 ; path B (shared tile) if r5-r3 < 24 and r8 < 24
## tile load (path B), thread t = tid.y*16+tid.x, for idx = t; idx < w*h; idx += 256
 x = idx % w + r3 ; y = idx / w + r4 ; cx = x > 0 ? min(P208.x + x, P216.x) : P208.x (same for y)   [P208 v2u32 origin, P216 v2u32 max]
 c = texelFetch(T200 color, cx, cy) ; rgb = max(c, 0)
 Y = .25R + .5G + .25B ; Co = .5R - .5B ; Cg = .5G - .25R - .25B          (YCoCg)
 L = f1 * Y ; k = L < 1000 ? 1/(L+1) : (BB127 const) ...
 k = L < 1000 ? 1/(L+1) : (log2(9L+1)*0.6931472/9 - 0.012676) / L          (range compression of the exposed luma)
 sharedColor[row*192 + col*8] = f16 { Y' = L*k, Co' = f1*Co*k, Cg' = f1*Cg*k, alpha = c.a }   (order in memory: Y', Co', Cg', a)
 depth d = texelFetch(T248, clamp like above with origin P96, max P104).r -> sharedDepth[row*96 + col*4] (f32)
 bar.sync
## main (after barrier), per output pixel (px,py) = 16*ctaid + tid
 token tx = int(px*P64.x)/2 (P64 = 0.5 -> px/4), sub = (int(py*.5)&1)*2 + (int(px*.5)&1) ; head H = buf P272, 40 f16 per token = 4 sub-blocks x 10
 a9 = H[(ty*P80 + tx)*40 + sub*10 + 9] ; x = .5*a9 ; f1800 = x<-3 ? 0 : x>3 ? 1 : .5 + .5*x(x^2+27)/(9x^2+27)     (sigmoid(a9))
 T240 motion, T224 history color, T232 hi-res history (2x)
 f292 = px+.5, f293 = py+.5 ; (ix,iy) = roundHalfAway(f292*P56 + P16)  (nearest render pixel) ; tile pos (ix-r3, iy-r4)
 depth dilation on the tile: d0 centre, diagonals Dmm(-1,-1) Dpm(row-1,col+1) Dmp(row+1,col-1) Dpp(+1,+1)
   use = d0 < (Dmm+Dpp)/2 - .001*d0  ||  d0 < (Dpm+Dmp)/2 - .001*d0
   pick the largest of d0, Dmm, Dpm(row-1,col+1), Dmp(row+1,col-1), Dpp in that order (strict >) -> (dx,dy); (0,0) if !use
 mv = texelFetch(T240, clampfn(P112 + (ix+dx, iy+dy)))   [clampfn(v): v > 0 ? min(v + P112, P120) : P112]
 q = (f292,f293) + P0 * (mv.xy + P8)      (previous position in output pixels; P0 = mv scale, P8 = mv offset)
 uvh = q * P144 ; uv2 = q * P152 ; p5 = P32(u8) != 0 || uv2.x < 0 || uv2.x > 1 (.. more below)
 hist = textureLod(T224, uvh)  -> f295..f298
 hi-res history (T232): h = (q-.5)*2+.5 ; base = floor(h-.5)+.5 ; t = sat(h-base) ; Catmull-Rom w0..w3(t) per axis
   (w0 = t^2-.5(t+t^3), w1 = 1.5t^3-2.5t^2+1, w3 = .5(t^3-t^2), w2 = 1-w0-w1-w3)
   gathers (textureGather .r) at uv = (P144*.5) * (base + {-.5, 1.5, 3.5}) in x and y ; each value v: v < .99 ? v : .99 + (v-.99)*.001
 f1858 = P172
 gather footprint: 5x5 hi-res texels T[c][r], c,r = 0..4, T[0][0] = texel (floor(h-.5)-1) i.e. gather(x0,y0).w ; all softclamped (.99 rule)
 a = w1/(w1+w2), b = 1-a (per axis), s = P172 (sharpness)
 row(r, c0) = a*T[c0+1][r] + b*T[c0+2][r] + s*( w0*sat(T[c0][r]) + (w1-a)*sat(T[c0+1][r]) + (w2-b)*sat(T[c0+2][r]) + w3*sat(T[c0+3][r]) )      (c0 = 0 or 1)
 col(c0, r0) = ay*row(r0+1,c0) + by*row(r0+2,c0) + s*( wy0*sat(row(r0,c0)) + (wy1-ay)*sat(row(r0+1,c0)) + (wy2-by)*sat(row(r0+2,c0)) + wy3*sat(row(r0+3,c0)) )
 Hs[sx][sy] = clamp(col(sx, sy), min/max of T[sx+1..sx+2][sy+1..sy+2])   for sx,sy in {0,1}: f300=(0,0) f301=(1,0) f302=(0,1) f303=(1,1)
 re-exposure of each: g(v) = v < .999 ? v/(1-v) : (exp2((v+.012676)*9*1.442695)-1)/9 ; Lh = f2*g(v) ; v' = Lh * k(Lh)   (k as in tile load)
   -> f314 (0,0), f322 (1,0), f330 (0,1), 4th from f331 in next block
 p6 = p5 || uv2.y < 0 || uv2.y > 1        (history invalid)
## BB162-164
 f342 = Hs'(1,1). nonfinite = !( |f314*f322 + f330*f342 + hist.r*hist.g + sig*hist.b| < inf ) ; p7 = p6 || nonfinite
 sig = sat(f1800) ; s1 = p7 ? 1 : sig ; ha = p7 ? 1 : hist.a ; f343 = s1*(1-ha) + ha ; f344 = ha*(1-s1^2) + s1^2 ; f345 = P176
 current colour, bilinear from the tile: u = (px+.5,py+.5)*P40 + P16 - (r3,r4) [to f16] ; i = floor(u) ; f = sat(u - i) (f16)
   texel words at tile[i.y*192 + i.x*8 (+8, +192, +200)], weights (1-fx)(1-fy), fx(1-fy), (1-fx)fy, fx*fy, f16 fma chain
   -> cur = (Yb, Cob, Cgb, ab) = (f2251, f2252, f2253, f2254)
 (a0,a1) = H[tok*40 + sub*10 + 0,1] ; bright = Yb > .8 ; quad = OR of bright over the 2x2 pixel quad (px^1, py^1)  [shfl bfly 16 then 1] = r255
 m0 = a0/(1+|a0|), m1 = a1/(1+|a1|)   (f16 softsign; rs606, rs607)
 Hq[4] = p7 ? 0 : Hs' ; hc = p7 ? 0 : hist.rgb
 f367 = max(f344/(f344+1), P176)
## BB166: kernel-predicted hi-res luma filter (4 sub-pixels of the output pixel, 12 taps of the tile's Y')
 i0 = clamp(roundEven(m0*P188)+8, 0, 15), i1 likewise with m1 (P188 f16 = 8.0)
 A = (sig < .25 && f343 < .25) ; B = (P180 > f343) ; idx = ((quad<<2 | A<<1 | B) << 8) | (i0<<4 | i1)
 LUT at buffer P280: 16-byte entries: f16 e0 (lo of word0), e1 (hi word0), e2 (lo word1), e3 (hi word1) = slope, u16 e4 (lo word2) = orientation, e5 (hi word2), e6 (lo word3)
 sub-pixel positions in tile coords: xs[j] = (2px+j+.5)*P48.x + P16.x - r3 (j=0,1), ys[j] likewise (f16)
 if e4 == 0: A-axis = y, B-axis = x else A = x, B = y.   cA, cB = pixel-centre tile coords (rs329/rs330 = u from the bilinear step)
 rows: A_r = floor(cA) - 1 + r, r = 0..3 ; centre line cB_r = cB + (A_0 - cA)*e3 + r*e3 ; B start = roundEven(cB_r) - 1 ; taps B, B+1, B+2
 for each sub-pixel s: dA = A_r - subA_s ; dB = B - subB_s ; E = e0*dA^2 + e2*dA*dB + e1*dB^2 (f16 fma chain) ; w = exp2(E)
   sumY += w * Y'(tap) ; sumW += w ; track the tap with the largest E (strict >, first wins) and its Y' (Ymax)
 F[s] = sumY/sumW (f16; f32 rcp with one refinement step for tiny quotients) -> rs608 (x0,y0), rs609 (x1,y0), rs610 (x0,y1), rs611 (x1,y1)
 Emax pairs r260 (subs 0,1), r261 (subs 2,3); Ymax pairs r262, r263
## BB174: blend with history, hi-res store
 Fsel[s] = (Emax[s] < -12) ? Ymax[s] : F[s]
 confidence per sub-pixel (f16): rA = roundEven(subA) ; dA = rA - subA ; line = subB + dA*e3 ; dB = |roundEven(line) - line|
   KA = (e4==0) ? P192 : P190 ; KB = (e4==0) ? P190 : P192      (f16 params at 190 / 192)
   cB = sat(dB*KB + 4.5) ; cA = sat(|dA|*KA + 4.5) ; conf[s] = (cA*e5 + e6) * cB ; avg = .25 * sum(conf)
 v = f16(f343) ; v2 = .1 + .9*v ; lo = (v - P184)*P186 ; hi = (v2 - P184)*P186     (P184, P186 f16 params)
 alphaC = sat(avg*(v2 - hi) + hi)  (f2349) ; bl[s] = sat(conf[s]*(v - lo) + lo)  (f2350..3)
 nonfinite2 = !(|Yb*Fsel0*Fsel1*Fsel2*Fsel3| < inf) -> cur and Fsel all 0
 N[s] = Hq[s] + bl[s]*(Fsel[s] - Hq[s])                               (new hi-res luma, f384..f387)
 store S296 (R16F 2x): texel (2px+sx, 2py+sy) = enc(N[s]), enc(v) = v < .99 ? v : .99 + (v-.99)*1000
 colour: C = alphaC*cur + (1-alphaC)*hc (Y,Co,Cg) ; rCo = C.Co/max(C.Y,1e-5), rCg likewise ; Nm = .25*sum(N) ; f391 = rCo*Nm ; f392 = rCg*Nm
 p9 = nonfinite || nonfinite2 ; P90(u8) == 0 -> BB203 (the path taken in the captures; P89, P90, P91 all 0). P90 != 0 path not analysed.
## tails (P90 == 0)
 BB203: (Yn, Con, Cgn) = (f390 = Nm, f391, f392) [debug colours if P89 && p9] ; P91 != 0 -> BB208 else BB205..207 -> final output S312
 BB210 history colour S288(px,py): Y = max(Nm,0) ; lim = Y ; if (Cgn - 1e-5 < -Y) lim = 1e-5 - Cgn ; if (Con - Cgn - 1e-5 < -lim) lim = Cgn - Con + 1e-5 ;
      if (-Con - Cgn - 1e-5 < -lim) lim = Con + Cgn + 1e-5 ; sc = (lim <= Y) ? 1 : Y/lim ; store (Y, Con*sc, Cgn*sc, sat(f367))
 BB213 S304 (token grid): only for px < P80 && py < P84 (px,py used as token coords): m_s = .2 * sum_{k=3..7} H[(py*P80+px)*40 + s*10 + k], s=0..3 ;
      n = rsqrt(sum m_s^2 + 1) ; store (m0,m1,m2,m3)*n
 final S312: if P91 == 0: (Y,Co,Cg) = (Yn,Con,Cgn) * ginv(Yn) / f1, ginv(v) = v < .999 ? 1/(1-v) : (exp2((v+.012676)*9*1.442695)-1)/(9v)
      RGB = (Y + Co - Cg, Y + Cg, Y - Co - Cg) ; if px < P136.x && py < P136.y: store S312(P128 + (px,py)) = (R, G, B, ab)

## display-resolution motion (frame flag 4, `D4RO0003`)
 Motion is read at the output pixel itself: mv = tex240(clampOrigin(px, P112, P120)), with no depth tile load and no
 nearest-depth selection (so depth is not read). P0 (scale) and P8 (offset) are used as before, with the motion
 texel in output pixels.
