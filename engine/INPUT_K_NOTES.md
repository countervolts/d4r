# K input kernel (hiluma_engine_input_depthinv_mvlo_hdr_v2_rel) - reverse-engineering notes
P = parameter block (248 bytes). Textures: 176 history colour (Y,Co,Cg,a compressed; linear), 184 motion, 192 depth, 200/208/216 exposure (1x1).
Filter modes are NGX's own (cuTexObjectCreate): 168, 176 and 224 linear, the others point; all clamped, normalized coordinates.
200 = current exposure, 208 = previous exposure, 216 = current exposure.
## entry
 a = tex208 (0 -> 1), b = tex200 (0 -> 1), c = tex216 (0 -> 1)
 f152 = P144*a ; f158 = P144*b ; f1 = P156*f158 (0 -> 1) ; f2 = (P148*c)/f158 ; f3 = f158/f152
 One thread per 2x2-pixel sub-block: gx = 16*ctaid.x + tid.x, gy = 16*ctaid.y + tid.y ;
   X = 4*(gx>>2) + 2*(tid.x&1), Y = 4*gy + (tid.x&2)   -> token (gx>>2, gy), sub index = (tid.x&1) + 2*((tid.x>>1)&1)
 f4 = X+.5, f5 = Y+.5 ; (ix,iy) = roundHalfAway((f4,f5)*P40 - .5 + P24)
 depth (tex192) at (ix,iy) and the four diagonal neighbours, each clamped as v > 0 ? min(P88 + v, P88 + P96) : P88
 same dilation rule as the output kernel -> (dx,dy) ; mv = texelFetch(tex184, clampfn(P104 + (ix+dx,iy+dy))), max = P104 + P112
 q = (f4,f5) + P8*(mv + P16) ; hist = textureLod(tex176, q*P120)   (f22..f25) ; P136 loaded (f201,f202)
## the four pixels j of the sub-block: (X,Y), (X+1,Y), (X,Y+1), (X+1,Y+1)
 each: nearest render texel, dilated motion mv_j (as above), q_j = pixel + .5 + P8*(mv_j + P16), hist_j = textureLod(tex176, q_j*P120)
   lin_j = hist_j.rgb * expandK(hist_j.r) * f3      -> shared W[j][tid.y*16+tid.x] (3 floats)
   local[j] = ((token centre in px) + P8*(mv_j + P16)) * P128, token centre = 4*(X/4 + .5), 4*(Y/4 + .5)
   invalid |= reset(P32 u8) || q_j*P136 outside [0,1]   (accumulated over the four pixels: p3)
 r32 = 1 - subY ; r44 = r32 + X ; r45 = subX + Y
## BB12: current colour statistics, 3x3 render texels around the sub-block centre
 (cx,cy) = roundHalfAway((X+1, Y+1)*P40 - .5 + P24) ; colour tex168, clamp origin P72, max P72+P80 ; f481 = P160
 per tap (order (-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)): rgb = f1*max(c,0) ; Y,Co,Cg as in the output kernel
   wA = max(1/(Y+1), P160) ; S += wA*(Y,Co,Cg)
   t = log2(12*max(Y,0)+1)*.6931472 / (max(Y,0)+1e-6) ; L = (Y,Co,Cg)*t ; running min and max of L over the taps
 after the 9 taps: m = S/9 ; avg = m / max(1 - m.Y, P160) ; A = avg * t(avg.Y)        (t as above: log compression)
 halfr = .5*(max - min) ; den = 8*(max - min) + 1          (per channel)
 for each j: Hl = lin_j * t(lin_j.Y) ; d = max(|A - Hl| - halfr, 0)/den ; e_j = min(2*d.Y + .5*d.Co + .5*d.Cg, 1)
 mismatch = .25 * sum_j e_j
 representative pixel of the sub-block: offset (1 - subY, subX), i.e. j' = 2*subX + (1 - subY)
   Yc = exposed luma (f1 * (.25R + .5G + .25B) of max(c,0)) of the colour texel nearest to that pixel
   feat = textureLod(tex224 (previous token feature, 4 ch), local[j'])
   if invalid: mismatch = 0, Yh = Yc, feat = 0 else Yh = lin_j'.Y
 featSum = sum of feat over the token's four threads (shfl bfly 3, then 1)
 tc = (f2*Yc) * max(1/(f2*Yc+1), P160) ; th = (f2*Yh) * max(1/(f2*Yh+1), P160)
 then a switch on the sub index
## output: 4 f16 per sub-block into buffer P240 at ((tokY*P64 + tokX)*16 + sub*4)   (16 channels per token: the network input)
 G(x) = x > .0031308 ? 1.055*x^(1/2.4) - .055 : 12.92*x      (sRGB transfer)
 out0 = G(sat(tc)) ; out1 = G(sat(th)) ; out2 = .25*featSum[sub] (non-finite -> 0) ; out3 = mismatch (non-finite -> 0)
 No barrier: the shared array is only per-thread scratch.

## display-resolution motion (frame flag 4, `D4RO0003`)
 Vectors are output-sized and already in output pixels (scale 1 by default). Each output pixel j of a token uses its own
 vector: mv_j = mvScale*(tex184(clamp(X+j)) + mvOffs), with no diagonal depth dilation and no render-texel lookup.
 The rest (q_j, history fetch, local terms) is unchanged. The package must carry `D4RO0003`.
