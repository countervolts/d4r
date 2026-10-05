#!/usr/bin/env python3
"""Oracle debug stages for cuda_dldn_engine_swin_dec0_kernel's 19 phases.

Each stage is a (file line after which the store block goes, registers in order) tuple for
kernels/tex/make_ptx.py's SWIN_DEBUG_STAGES, so D4R_RRSWIN_DEBUG=p1 (p2, p3, ...) builds
NVIDIA's own body with that stage's registers stored into the buffer at arg048 + 46,000,000,
which kernels/rr/rrswin_dec0.hip's D4R_RRSWIN_DEBUG_STAGE reproduces from its own registers.

The registers are read out of ~/.cache/d4r-rr-corpus/0023-PREPASS_ENTRYPOINT_NAME.ptx with
kernels/rr/rr_layer_spec.py's own parser: a phase is a maximal run of adjacent mma, and a
stage keeps the mma whose D pair no other mma of the same phase reads as its C (the last
k-step of an accumulating chain) and lists those tiles in mma statement order, each as the
(row g, row g+8) pair the mma writes.  Phase 1 is 48x128 with K=64, so its 96 mma are 48
chains of two steps and it contributes 48 pairs; phase 5 and phase 10 are 64x32 with
K=96, 48 mma in 16 chains of three steps, and contribute 16 pairs each.  The splice line is
the `;` that ends the phase's last mma (its first line plus three).
"""


def swin_quads(bases):
    """One A fragment per base: four b32."""
    return sum(([f"%r{b + o}" for o in range(4)] for b in bases), [])


def swin_seq(regs):
    """A plain register list."""
    return list(regs)


def swin_pairs(bases):
    """One register pair per base: a tile's (row g, row g+8) D registers."""
    return sum(([f"%r{b + rh}" for rh in (0, 1)] for b in bases), [])


def stages():
    """The stage table: name -> (splice line, registers)."""
    return {
        # phase 1: 48 tiles, splice after file line 2061
        "p1": (2061,
            swin_pairs([484, 494, 524, 534, 564, 574, 604, 614, 644, 654, 684, 694, 724, 734, 764,
            774, 804, 814, 844, 854, 884, 894, 924, 934, 964, 974, 1004, 1014, 1044, 1054, 1084,
            1094, 1124, 1134, 1164, 1174, 1204, 1214, 1244, 1254, 1284, 1294, 1324, 1334, 1364,
            1374, 1404, 1414])),
        # phase 2: 24 tiles, splice after file line 10622
        "p2": (10622,
            swin_pairs([4655, 4665, 4675, 4685, 4695, 4705, 4715, 4725, 4735, 4745, 4755, 4765,
            4775, 4785, 4795, 4805, 4815, 4825, 4835, 4845, 4855, 4865, 4875, 4885])),
        # phase 3: 24 tiles, splice after file line 10803
        "p3": (10803,
            swin_pairs([4904, 4914, 4924, 4934, 4944, 4954, 4964, 4974, 4984, 4994, 5004, 5014,
            5024, 5034, 5044, 5054, 5064, 5074, 5084, 5094, 5104, 5114, 5124, 5134])),
        # phase 4: 48 tiles, splice after file line 11520
        "p4": (11520,
            swin_pairs([5321, 5331, 5341, 5351, 5361, 5371, 5381, 5391, 5401, 5411, 5421, 5431,
            5441, 5451, 5461, 5471, 5481, 5491, 5501, 5511, 5521, 5531, 5541, 5551, 5561, 5571,
            5581, 5591, 5601, 5611, 5621, 5631, 5641, 5651, 5661, 5671, 5681, 5691, 5701, 5711,
            5721, 5731, 5741, 5751, 5761, 5771, 5781, 5791])),
        # phase 5: 16 tiles, splice after file line 21999
        "p5": (21999,
            swin_pairs([8607, 8617, 8667, 8677, 8727, 8737, 8787, 8797, 8847, 8857, 8907, 8917,
            8967, 8977, 9027, 9037])),
        # phase 6: 16 tiles, splice after file line 22535
        "p6": (22535,
            swin_pairs([9089, 9099, 9109, 9119, 9129, 9139, 9149, 9159, 9169, 9179, 9189, 9199,
            9209, 9219, 9229, 9239])),
        # phase 7: 24 tiles, splice after file line 22716
        "p7": (22716,
            swin_pairs([9258, 9268, 9278, 9288, 9298, 9308, 9318, 9328, 9338, 9348, 9358, 9368,
            9378, 9388, 9398, 9408, 9418, 9428, 9438, 9448, 9458, 9468, 9478, 9488])),
        # phase 8: 24 tiles, splice after file line 22897
        "p8": (22897,
            swin_pairs([9507, 9517, 9527, 9537, 9547, 9557, 9567, 9577, 9587, 9597, 9607, 9617,
            9627, 9637, 9647, 9657, 9667, 9677, 9687, 9697, 9707, 9717, 9727, 9737])),
        # phase 9: 48 tiles, splice after file line 23614
        "p9": (23614,
            swin_pairs([9924, 9934, 9944, 9954, 9964, 9974, 9984, 9994, 10004, 10014, 10024, 10034,
            10044, 10054, 10064, 10074, 10084, 10094, 10104, 10114, 10124, 10134, 10144, 10154,
            10164, 10174, 10184, 10194, 10204, 10214, 10224, 10234, 10244, 10254, 10264, 10274,
            10284, 10294, 10304, 10314, 10324, 10334, 10344, 10354, 10364, 10374, 10384, 10394])),
        # phase 10: 16 tiles, splice after file line 34088
        "p10": (34088,
            swin_pairs([13210, 13220, 13270, 13280, 13330, 13340, 13390, 13400, 13450, 13460, 13510,
            13520, 13570, 13580, 13630, 13640])),
        # phase 11: 16 tiles, splice after file line 34325
        "p11": (34325,
            swin_pairs([13691, 13701, 13711, 13721, 13731, 13741, 13751, 13761, 13771, 13781, 13791,
            13801, 13811, 13821, 13831, 13841])),
        # phase 12: 16 tiles, splice after file line 36579
        "p12": (36579,
            swin_pairs([14179, 14189, 14199, 14209, 14219, 14229, 14239, 14249, 14259, 14269, 14279,
            14289, 14299, 14309, 14319, 14329])),
        # phase 13: 16 tiles, splice after file line 39042
        "p13": (39042,
            swin_pairs([15310, 15320, 15330, 15340, 15350, 15360, 15370, 15380, 15390, 15400, 15410,
            15420, 15430, 15440, 15450, 15460])),
        # phase 14: 16 tiles, splice after file line 39290
        "p14": (39290,
            swin_pairs([15512, 15522, 15532, 15542, 15552, 15562, 15572, 15582, 15592, 15602, 15612,
            15622, 15632, 15642, 15652, 15662])),
        # phase 15: 16 tiles, splice after file line 41450
        "p15": (41450,
            swin_pairs([16642, 16652, 16662, 16672, 16682, 16692, 16702, 16712, 16722, 16732, 16742,
            16752, 16762, 16772, 16782, 16792])),
        # phase 16: 16 tiles, splice after file line 41698
        "p16": (41698,
            swin_pairs([16844, 16854, 16864, 16874, 16884, 16894, 16904, 16914, 16924, 16934, 16944,
            16954, 16964, 16974, 16984, 16994])),
        # phase 17: 16 tiles, splice after file line 43858
        "p17": (43858,
            swin_pairs([17974, 17984, 17994, 18004, 18014, 18024, 18034, 18044, 18054, 18064, 18074,
            18084, 18094, 18104, 18114, 18124])),
        # phase 18: 16 tiles, splice after file line 44106
        "p18": (44106,
            swin_pairs([18176, 18186, 18196, 18206, 18216, 18226, 18236, 18246, 18256, 18266, 18276,
            18286, 18296, 18306, 18316, 18326])),
        # phase 19: 16 tiles, splice after file line 46266
        "p19": (46266,
            swin_pairs([19306, 19316, 19326, 19336, 19346, 19356, 19366, 19376, 19386, 19396, 19406,
            19416, 19426, 19436, 19446, 19456])),
        # E1: the dequantised skip input (24 v2.u16 loads at param_0+24, 48 e4m3x2 registers)
        "sk": (6695, swin_seq(list(range(4074, 4122)))),
        # E1: the dequantised patch read back from the shared window (the 24 ld.shared.v2.u16)
        "rp": (6840, swin_seq(list(range(4122, 4170)))),
        # E1: the 24 packed A registers phases 2 and 3 read, fragment by fragment
        "a2": (10442, swin_quads([9539, 9579, 9619, 9659, 9699, 9739])),
    }
