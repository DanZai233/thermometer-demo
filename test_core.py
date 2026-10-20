import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import core

class EvidenceTests(unittest.TestCase):
    def test_original_preserved_and_duplicates_flagged(self):
        b = io.BytesIO()
        Image.new('RGB', (32,32), 'red').save(b, format='PNG')
        raw = b.getvalue()
        with tempfile.TemporaryDirectory() as d, patch.object(core, 'DATA', Path(d)):
            a = core.ingest(raw, 'first.png')
            b = core.ingest(raw, 'second.png')
            self.assertEqual(Path(a['path']).read_bytes(), raw)
            for item in [a,b]:
                item.update(stage='维修前', facts={'temperature':20.8, 'readability':'clear'})
            self.assertTrue(any('重复上传' in s for s in core.checks([a,b], [])))
            with patch.dict('os.environ', {}, clear=True), patch('local_models.readiness', return_value={'ai':False,'local':False}):
                self.assertEqual(core.forensic(a)['status'], '未运行')

    def test_scene_conflicts_not_authenticity_verdict(self):
        a = {'name':'A', 'sha256':'a', 'phash':'0000000000000000', 'stage':'维修前',
             'facts':{'temperature':38.1,'measurement':'surface','readability':'clear', 'watermark_date':'2025-12-02'}}
        b = {'name':'B', 'sha256':'b', 'phash':'ffffffffffffffff', 'stage':'维修后',
             'facts':{'temperature':None,'measurement':'unknown','watermark_date':'2026-01-29'}}
        signals = core.checks([a,b], [])
        self.assertTrue(any('表面' in s for s in signals))
        self.assertTrue(any('日期不同' in s for s in signals))
        self.assertTrue(any('读数未确认' in s for s in signals))

if __name__ == '__main__':
    unittest.main()
