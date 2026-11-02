#!/usr/bin/env python3
"""Print the streaming (SPM) counters of an RGP capture as a timeline.

profile_rr.py captures with counter collection; the derived counters RGP shows
in its Counters view are stored in the capture as ``DerivedSpmCtr`` chunks (one
sample per ``sample_frequency`` GPU cycles).  This reads them without the GUI.
The capture covers the gated stage's submission, so consecutive dispatches show
up as plateaus in the timeline (a dip marks the barrier between them); pass the
plateau's time range to ``--window`` for per-dispatch totals.  Percent counters
are averaged, count counters summed.  Requires an uncompressed RDF v3 capture.
"""
import argparse
import csv
import math
from pathlib import Path
import struct

KEY = ('Gpu Busy Cycles', 'Memory unit busy', 'Memory unit stalled', 'LDS Bank Conflict',
       'Instruction cache hit', 'L0 cache hit', 'L2 cache hit', 'Fetch size', 'Write size')


def read(capture):
    blob = capture.read_bytes()
    if blob[:8] != b'AMD_RDF ' or struct.unpack_from('<I', blob, 8)[0] != 3:
        raise ValueError('not an RDF v3 capture')
    offset, size = struct.unpack_from('<QQ', blob, 16)
    entries = [struct.unpack_from('<16sIIQQQQQ', blob, i) for i in range(offset, offset + size, 64)]

    def named(name):
        return [e for e in entries if e[0].rstrip(b'\0') == name.encode()]

    def payload(entry):
        if entry[1] != 0:
            raise ValueError('compressed chunk; save the capture uncompressed')
        return blob[entry[5]:entry[5] + entry[6]]

    session = named('SpmSession')[0]
    _, _, frequency, count, _ = struct.unpack_from('<5I', blob, session[3])
    times = struct.unpack('<%dQ' % count, payload(session))
    counters = []
    for entry in named('DerivedSpmCtr'):
        _, index, components, unit, name_size, desc_size = struct.unpack_from('<6I', blob, entry[3])
        data = payload(entry)
        values = struct.unpack_from('<%dd' % count, data)
        at = count * 8 + components * 4
        name = data[at:at + name_size].decode().rstrip('\0')
        description = data[at + name_size:at + name_size + desc_size].decode().rstrip('\0')
        counters.append((index, name, '%' in description[:80] or 'percentage' in description.lower(), values))
    return frequency, times, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--bins', type=int, default=32, help='timeline rows')
    parser.add_argument('--window', type=float, nargs=2, metavar=('START', 'END'),
                        help='summarize samples in [START, END) (timeline time units)')
    parser.add_argument('--csv', type=Path, help='write every sample of every counter')
    args = parser.parse_args()
    frequency, times, counters = read(args.capture)
    t0 = times[0]
    if args.csv:
        with args.csv.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time', *('%s#%d' % (name, index) for index, name, _, _ in counters)])
            writer.writerows(zip((t - t0 for t in times), *(c[3] for c in counters)))
    key = [c for c in counters if c[1] in KEY]
    print('%d samples, one per %d cycles' % (len(times), frequency))
    if args.window:
        rows = [i for i, t in enumerate(times) if args.window[0] <= t - t0 < args.window[1]]
        print('window: %d samples (%d cycles)' % (len(rows), len(rows) * frequency))
        for index, name, percent, values in counters:
            v = [values[i] for i in rows if math.isfinite(values[i])]
            if v:
                print('%3d %-28s %s %.4g' % (index, name, 'mean' if percent else 'sum ',
                                             sum(v) / len(v) if percent else sum(v)))
        return
    step = max(1, len(times) // args.bins)
    print('%9s ' % 'time' + ' '.join('%11.11s' % c[1] for c in key))
    for i in range(0, len(times), step):
        span = range(i, min(i + step, len(times)))
        cells = []
        for _, _, _, values in key:
            v = [values[j] for j in span if math.isfinite(values[j])]
            cells.append('%11.1f' % (sum(v) / len(v) if v else float('nan')))
        print('%9d ' % (times[i] - t0) + ' '.join(cells))


if __name__ == '__main__':
    main()
