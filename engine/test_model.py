"""Structural model corruption must be rejected before compiling/dispatching shaders."""
import unittest
from compile_k import validate
from extract_k import LAYERS


def minimal_index():
    index = {}
    for name, (kind, h, c, x, pos) in LAYERS.items():
        e = {'kind': kind, 'H': h, 'C': c, 'X': x, 'posattn': pos,
             'g1': 0, 'bo': 0, 'g2': 0, 'b2': 0,
             'qkv': [0] * (h * 3), 'table': [0] * h, 'wo': [0] * h,
             'w1': [0] * (c // 8), 'b1': [0] * (c // 8), 'w2': [0] * (c // 8)}
        if kind in ('enc', 'enc0'): e.update(merge_n=x, merge_w=0, merge_b=0)
        if kind in ('dec', 'dec0'): e.update(expand_w=0, expand_b=0)
        if kind == 'enc0': e.update(embed_w=0, embed_b=0)
        if kind == 'dec0': e.update(head_w=0, head_b=0)
        index[name] = e
    return index


class ModelTests(unittest.TestCase):
    def setUp(self): self.index = minimal_index()
    def test_shared_zero_weights_are_structurally_valid(self):
        validate(self.index, 4 * 1024 * 1024)
    def test_matrix_end_past_blob_is_rejected(self):
        self.index['dec4']['expand_w'] = 4 * 1024 * 1024 // 2 - 16
        with self.assertRaisesRegex(ValueError, 'outside'): validate(self.index, 4 * 1024 * 1024)
    def test_weight_alignment_is_checked(self):
        self.index['enc0']['embed_w'] = 1
        with self.assertRaisesRegex(ValueError, 'alignment'): validate(self.index, 4 * 1024 * 1024)
    def test_wrong_network_architecture_is_rejected(self):
        self.index['enc3']['H'] = 2
        with self.assertRaisesRegex(ValueError, 'shape'): validate(self.index, 4 * 1024 * 1024)
    def test_missing_merge_is_rejected(self):
        del self.index['enc2']['merge_w']
        with self.assertRaisesRegex(ValueError, 'missing'): validate(self.index, 4 * 1024 * 1024)
    def test_incomplete_head_set_is_rejected(self):
        self.index['dec5']['qkv'].pop()
        with self.assertRaisesRegex(ValueError, 'expected'): validate(self.index, 4 * 1024 * 1024)
    def test_wrong_layer_order_is_rejected(self):
        index = dict(reversed(list(self.index.items())))
        with self.assertRaisesRegex(ValueError, 'order'): validate(index, 4 * 1024 * 1024)
    def test_odd_blob_bytes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'f16'): validate(self.index, 4 * 1024 * 1024 - 1)
    def test_invalid_merge_stride_is_rejected(self):
        self.index['enc2']['merge_n'] = 128
        with self.assertRaisesRegex(ValueError, 'stride'): validate(self.index, 4 * 1024 * 1024)


if __name__ == '__main__': unittest.main()
