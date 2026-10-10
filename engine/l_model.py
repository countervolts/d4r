#!/usr/bin/env python3
"""Preset L (rrlite, unfolded): plain weights of the input stage (rrlite_enc0_4x4_*) and the expansion stage
(rrlite_dec0_4x4), the two stages L does not share with preset M. See L_NOTES.md.

Both are a Swin block of preset M's shape at render-pixel resolution (C 32, 2 heads, 4x4-pixel windows); their weight
allocations are laid out like a network layer's, so m_model.plain_layer reads them:
  enc0: the dither table (4096 bytes), the embedding (W1 [32][32] FP8, b1, W2 [32][32] FP8, b2) at 4096, then the
        block and its patch merge (128 -> 64 channels) at 6272
  dec0: the patch expand (64 -> 4 x 32 channels) and its bias, then the block (m_model.plain_layer with CIN 64)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m_model as mm
import model_enc3 as me
import swin_model as sm

E_EMB, E_BLOCK = 4096, 6272
mm.LAYERS.update({'enc0l': (32, 2, 64, 0), 'dec0l': (32, 2, 0, 64)})


class Blob:
    """the part of a weight allocation that m_model.plain_layer reads (model_enc3.Params' wbuf and f16vec)"""
    def __init__(self, wbuf):
        self.wbuf = wbuf

    def f16vec(self, off, n):
        return self.wbuf[off:off + 2 * n].view(np.float16).copy()


def enc0_layer(wbuf):
    """plain tensors of the input stage: the block, its merge and the embedding (emb_*), natural channel order"""
    t = mm.plain_layer(Blob(wbuf[E_BLOCK:]), 'enc0l')
    fp8 = lambda T: me.E4M3[wbuf[T]]
    b = Blob(wbuf)
    g = mm.gp(32)
    # the embedding has the layout of an MLP chunk of a C 32 block: fc1's columns in K-slot order, fc2's rows in pair order
    t['emb_w1'] = np.asarray(fp8(sm.woff_table(E_EMB, 512, 16 * 32, 32, 32)), np.float16)
    t['emb_b1'] = b.f16vec(E_EMB + 32 * 32, 32)
    raw = fp8(sm.woff_table(E_EMB + 32 * 32 + 64, 0, 512, 32, 32))
    w2 = np.zeros((32, 32)); w2[g] = raw[:, g]
    t['emb_w2'] = np.asarray(w2, np.float16)
    t['emb_b2'] = b.f16vec(E_EMB + 2 * 32 * 32 + 64, 32)[g]
    return t


def dec0_layer(wbuf):
    """plain tensors of the expansion stage: the patch expand and the block"""
    return mm.plain_layer(Blob(wbuf), 'dec0l')
