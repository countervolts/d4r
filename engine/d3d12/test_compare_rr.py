"""RR acceptance compares two losses to one reference, with HDR RGB metrics."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from compare_rr import compare, read


class ReferenceQuality(unittest.TestCase):
    def test_pooled_hdr_rgb_and_backend_loss_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('case','reference','linux','dx'): (root/name).mkdir()
            (root/'case/frames.txt').write_text('size 2 2 2 2\nframe 0\nframe 1\n')
            for frame,error in ((0,.125),(1,.5)):
                ref=np.full((4,4),8,np.float16)
                # Alpha is not a quality channel; a nonfinite alpha must not
                # conceal or invalidate the RGB comparison.
                ref[:,3]=np.nan
                ref.tofile(root/f'reference/{frame}-out.bin')
                for name,scale in [('linux',1),('dx',1)]:
                    value=ref.copy();value[:,:3]+=error*scale;value.tofile(root/f'{name}/{frame}-out.bin')
            result=compare(root/'case',root/'reference',root/'linux',root/'dx')
            expected=-10*math.log10((.125**2+.5**2)/2)
            self.assertAlmostEqual(result['backends']['linux']['pooled_psnr_db'],expected)
            self.assertTrue(result['passed']);self.assertEqual(result['extra_loss_db'],0)
            value=np.full((4,4),9,np.float16)
            for frame in (0,1):value.tofile(root/f'dx/{frame}-out.bin')
            self.assertFalse(compare(root/'case',root/'reference',root/'linux',root/'dx')['passed'])

    def test_missing_shape_and_nonfinite_rgb_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'frame.bin'
            np.zeros(3,np.float16).tofile(p)
            with self.assertRaises(ValueError):read(p,1)
            np.array([0,np.inf,0,1],np.float16).tofile(p)
            with self.assertRaises(ValueError):read(p,1)
            with self.assertRaises(FileNotFoundError):read(p.with_name('missing.bin'),1)


if __name__=='__main__':unittest.main()
