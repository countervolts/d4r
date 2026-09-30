"""Compare compile-time Swin diagnostic snapshots with the NumPy intermediate values."""
import argparse
import numpy as np
from m_replay_validate import me, sm, LAYERS, metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture-dir', required=True)
parser.add_argument('--stages', required=True)
parser.add_argument('--block', default='0,4')
parser.add_argument('--layer', choices=list(LAYERS), default='enc1')
args = parser.parse_args()
p = me.Params(args.capture_dir)
c, heads, _, cin = LAYERS[args.layer]
bx, by = map(int, args.block.split(','))
pre = 4 * c * cin + 8 * c if cin else 0
x0 = sm.decoder_input(p, bx, by, c, cin, heads, p.p32) if cin else sm.load_tokens(p, bx, by, c // 32)
stages = {}
sm.swin_block(p, x0, c, heads, pre, stages=stages)
actual = np.fromfile(args.stages, np.float16).reshape(4, 64, 128).astype(np.float64)
for index, name in enumerate(('h1', 'x1', 'h2', 'O')):
    expected = stages[name]
    values = actual[index, :, :expected.shape[1]]
    metrics(expected, values, f'SWIN_STAGE {name}')
    different = np.argwhere(expected != values)
    for row, channel in different[:8]:
        print(f'DIFFERENT stage={name} token={row} channel={channel} reference={expected[row, channel]} native={values[row, channel]}')
