"""Reject incomplete GPU timings and exclude each input's warmup samples."""
import unittest
from bench_rr import STAGES, samples


def row(frame, repeat, total=100):
    return f'frame {frame} repeat {repeat} us {total} stages ' + ' '.join(['1'] * len(STAGES)) + '\n'


class TimingReplay(unittest.TestCase):
    def test_warmup_is_per_input(self):
        log = row(7, 0, 900) + row(7, 1, 100) + row(9, 0, 800) + row(9, 1, 200)
        self.assertEqual([r[0] for r in samples(log, [7, 9], 2, 1)], [100, 200])

    def test_incomplete_duplicate_or_invalid_samples_fail(self):
        for log in (row(7, 0), row(7, 0) * 2, row(7, 0, 0), row(7, 0, 'nan')):
            with self.subTest(log=log), self.assertRaises(ValueError):
                samples(log, [7], 2, 1)
        with self.assertRaises(ValueError):
            samples(row(7, 0) + row(7, 1).replace('1 1', '-1 1', 1), [7], 2, 1)


if __name__ == '__main__':
    unittest.main()
