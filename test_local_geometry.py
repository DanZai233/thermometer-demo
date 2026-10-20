import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from local_models import jpeg_features, region_evidence, tile_starts

class LocalEvidenceGeometry(unittest.TestCase):
    def test_jpeg_coefficients_preserved_on_nonmultiple_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'odd.jpg'
            Image.new('RGB',(539,617),'gray').save(p,quality=91)
            rgb,dct,qt,converted=jpeg_features(p)
            self.assertFalse(converted)
            self.assertEqual(rgb.shape,(617,539,3))
            self.assertEqual(dct.shape,(624,544))
            self.assertEqual(qt.shape,(8,8))
            for length in [539,617,2844]:
                covered=np.zeros(length,bool)
                for start in tile_starts(length):
                    self.assertEqual(start%8,0)
                    covered[start:min(length,start+512)]=True
                self.assertTrue(covered.all())

    def test_roi_uses_fixed_scores_not_visual_color(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scores.npy'
            a=np.zeros((100,200),np.float32);a[20:40,60:100]=.9
            np.save(p,a)
            result={'scores_path':str(p)}
            inside=region_evidence(result,[.3,.2,.5,.4])
            outside=region_evidence(result,[0,0,.2,.2])
            self.assertEqual(inside['high_response_fraction'],1)
            self.assertEqual(outside['high_response_fraction'],0)

if __name__=='__main__':unittest.main()
