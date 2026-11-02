#!/usr/bin/env python3
"""Pack the CNN Ray Reconstruction model's weights (nvngx_dlssd.dll 310.3, presets A-C) for engine/rr/cnn_*.comp.

NVIDIA's layouts (binary16): a 3x3 conv [K-chunk][tap][out][16], a 1x1 conv [K-chunk][out][16] (K-chunk = 16 input
channels; the head has 16 outputs of which 12 are used). The native shaders run every 16-deep step as one WMMA, and
the WMMA's f32 sum depends on the order of its 16 products: ZLUDA's gfx12 lowering of NVIDIA's mma.m16n8k16 puts
NVIDIA's k = P16[s] in WMMA K slot s (zluda_ptx_impl.cpp, d4r_wmma12_*). So every 16-block of K and of the outputs is
stored permuted by P16 here: a shader's output column s of a 16-block is channel P16[s], which is what the next
conv's K slot s must hold.
"""
import numpy as np

P16 = np.array([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15])


def perm(n):
    """position -> channel for n channels: P16 within each 16-block"""
    return (np.arange(n) // 16) * 16 + P16[np.arange(n) % 16]


def pack3x3(w, cin, cout):
    """[kc][tap][out][16] -> [kc][tap][out permuted][k permuted]"""
    w = np.asarray(w, np.float16)[:9 * cin * cout].reshape(cin // 16, 9, cout, 16)
    return np.ascontiguousarray(w[:, :, perm(cout)][..., P16])


def pack1x1(w, cin, cout, block=None):
    """[kc][out][16] -> [kc][out permuted][k permuted]. block: NVIDIA's layout is [out-block][kc][block][16] (the
    levels whose warps compute 32 outputs each loop over the output blocks: Enc2-4's middle conv)"""
    w = np.asarray(w, np.float16)[:cin * cout]
    if block:
        w = w.reshape(cout // block, cin // 16, block, 16).transpose(1, 0, 2, 3)
    w = w.reshape(cin // 16, cout, 16)
    return np.ascontiguousarray(w[:, perm(cout)][..., P16])
