import struct
import unittest

from rr_hlsl import vectorize_attention_bias


SOURCE = '''#define D4R_THREADS 128
static const uint HEAD = 17920u - 2560u;
[loop] for (uint hd = 0u; hd < (2u + rrB0.Load(1032u)); ++hd) {
  RRHalf rrW0_[6];
  [unroll] for (uint kt = 0u; kt < 6u; kt += 1u) {
    [unroll] for (uint ri_ = 0u; ri_ < 8u; ++ ri_) rrW0_[(kt - (0u)) / (1u)][ri_] = rrB1.Load<float16_t>(((((64u + HEAD * hd) / 2u + kt * 1024u + 16u * wv) + (ri_ + 8u * (WaveGetLaneIndex() >> 4u)) * (64u) + (WaveGetLaneIndex() & 15u))) * 2u);
  }
}
'''


class AttentionBiasTests(unittest.TestCase):
    def setUp(self):
        self.weights = struct.pack('<16384H', *range(16384))

    def test_every_lane_keeps_original_half_bits(self):
        _, packed = vectorize_attention_bias(SOURCE, self.weights)
        for head in range(2):
            for tile in range(6):
                for wave in range(4):
                    for lane in range(32):
                        at = ((head * 24 + tile * 4 + wave) * 256 +
                              (lane & 15) * 16 + (lane >> 4) * 8) * 2
                        actual = struct.unpack_from('<8H', packed, at)
                        expected = tuple(32 + head * 7680 + tile * 1024 +
                                         ((lane >> 4) * 8 + element) * 64 +
                                         wave * 16 + (lane & 15) for element in range(8))
                        self.assertEqual(actual, expected)

    def test_rejects_unknown_layout_and_truncated_weights(self):
        with self.assertRaises(ValueError):
            vectorize_attention_bias(SOURCE.replace('* (64u)', '* (32u)'), self.weights)
        with self.assertRaises(ValueError):
            vectorize_attention_bias(SOURCE, self.weights[:20000])



if __name__ == '__main__':
    unittest.main()
