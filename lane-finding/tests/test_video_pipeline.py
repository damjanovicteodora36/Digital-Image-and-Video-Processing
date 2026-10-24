import tempfile
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import cv2
import numpy as np
from src.video_pipeline import process_frame,make_context,process_video,video_info,verify_video


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.context=make_context()

    def test_empty_scene_and_independence(self):
        image=np.zeros((720,1280,3),np.uint8)
        result,data=process_frame(image,self.context)
        self.assertEqual(result.shape,image.shape)
        self.assertEqual(result.dtype,np.uint8)
        self.assertFalse(data['lane_rendered'])
        self.assertIsNone(data['geometry']['offset_m'])
        again,second=process_frame(image,self.context)
        np.testing.assert_array_equal(result,again)
        self.assertEqual(data,second)
        self.assertFalse(image.any())

    def test_invalid_frames(self):
        for frame in [np.zeros((540,960,3),np.uint8),np.zeros((720,1280),np.uint8),np.zeros((720,1280,3),float),None]:
            with self.assertRaises(ValueError): process_frame(frame,self.context)

    def test_real_image(self):
        root=Path(__file__).resolve().parents[1]
        image=cv2.imread(str(root/'test_images/straight_lines1.jpg'))
        result,data=process_frame(image,self.context)
        self.assertEqual(result.shape,image.shape)
        self.assertTrue(data['lane_rendered'])

    def test_one_unreliable_boundary(self):
        root=Path(__file__).resolve().parents[1]
        image=cv2.imread(str(root/'test_images/test1.jpg'))
        result,data=process_frame(image,self.context)
        self.assertTrue(data['fits']['left']['reliable'])
        self.assertFalse(data['fits']['right']['reliable'])
        self.assertFalse(data['lane_rendered'])
        self.assertIsNone(data['geometry']['offset_m'])

    def test_video_roundtrip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'input.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,720))
            self.assertTrue(writer.isOpened())
            for _ in range(3): writer.write(np.zeros((720,1280,3),np.uint8))
            writer.release()
            report=process_video(source,root/'output')
            self.assertEqual(report['output']['decoded_frames'],3)
            self.assertEqual(report['longest_failure_frames'],3)
            # Exercise nonempty adjacent deltas and JSON serialization.
            calls=iter([640.0,680.0,640.0])
            def fake_frame(frame,context):
                return frame,dict(two_accepted_boundaries=True,lane_rendered=True,
                    geometry=dict(lane_center_x_px=next(calls),reasons=[]),warnings=[],
                    initialization=[dict(method='primary'),dict(method='primary')])
            with patch('src.video_pipeline.process_frame',side_effect=fake_frame):
                changed=process_video(source,root/'jumps')
            self.assertEqual(changed['adjacent_center_change']['over_30_px'],2)
            self.assertEqual(changed['adjacent_center_change']['max_absolute_px'],40)
            self.assertEqual(json.loads((root/'jumps/report.json').read_text())['total_frames'],3)
            with self.assertRaises(OSError): video_info(root/'missing.mp4')
            with self.assertRaises(OSError): verify_video(source,dict(width=1280,height=720,fps=25,frames=4))

    def test_incompatible_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'small.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(800,600))
            writer.write(np.zeros((600,800,3),np.uint8)); writer.release()
            with self.assertRaises(ValueError): process_video(source,Path(temporary)/'output')


if __name__=='__main__': unittest.main()
