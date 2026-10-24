"""Resolution configuration, shared calibration and safe final-video output."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from src.thresholding import load_calibration
from src.video_config import resolution_config,scale_calibration
from src.video_pipeline import make_context,process_frame,process_video

class VideoResolutionTests(unittest.TestCase):
    def test_tested_thresholds_and_points(self):
        large=resolution_config((1280,720));small=resolution_config((960,540))
        self.assertEqual(large['search']['margin'],80);self.assertEqual(large['fit']['min_pixels'],300)
        self.assertEqual(small['search']['margin'],60);self.assertEqual(small['search']['minpix'],29)
        self.assertEqual(small['fit']['min_pixels'],169);self.assertEqual(small['fit']['min_unique_y'],75)
        self.assertEqual(small['search']['seed_parameters'],dict(min_peak=15,peak_distance=45))
        self.assertEqual(small['search']['min_supported_windows'],4)
        self.assertEqual(small['destination_points'][2][1],539)
        self.assertEqual(small['geometry']['max_width_variation_fraction'],.25)
        self.assertFalse(small['temporal_tracking'])

    def test_scaled_camera_and_same_distortion(self):
        original=load_calibration();large=make_context();small=make_context((960,540))
        np.testing.assert_array_equal(large[0][0],original[0])
        np.testing.assert_array_equal(small[0][0],np.diag([.75,.75,1])@original[0])
        np.testing.assert_array_equal(small[0][1],original[1]);self.assertEqual(small[0][2],(960,540))
        np.testing.assert_array_equal(load_calibration()[0],original[0])

    def test_small_frame_independence_and_dimensions(self):
        image=np.zeros((540,960,3),np.uint8);ctx=make_context((960,540))
        a,da=process_frame(image,ctx);b,db=process_frame(image,ctx)
        self.assertEqual(a.shape,image.shape);self.assertEqual(a.dtype,np.uint8)
        np.testing.assert_array_equal(a,b);self.assertEqual(da,db)
        self.assertTrue(da['undistortion_applied']);self.assertTrue(da['camera_identity_assumed'])
        self.assertFalse(da['camera_identity_independently_confirmed'])
        self.assertFalse(da['lane_rendered']);self.assertFalse(image.any())

    def test_automatic_resolution_and_real_resized_input(self):
        root=Path(__file__).resolve().parents[1]
        image=cv2.resize(cv2.imread(str(root/'test_images/straight_lines1.jpg')),(960,540))
        a,da=process_frame(image);b,db=process_frame(image,make_context((960,540)))
        np.testing.assert_array_equal(a,b);self.assertEqual(da,db)

    def test_unsupported_resolution_and_context_mismatch(self):
        with self.assertRaises(ValueError):resolution_config((800,600))
        with self.assertRaises(ValueError):scale_calibration(load_calibration(),(1920,1080))
        with self.assertRaises(ValueError):process_frame(np.zeros((540,960,3),np.uint8),make_context())

    def test_small_video_roundtrip_and_scaled_jump_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(960,540))
            self.assertTrue(writer.isOpened())
            for _ in range(3):writer.write(np.zeros((540,960,3),np.uint8))
            writer.release()
            report=process_video(source,root/'actual')
            self.assertEqual(report['output']['decoded_frames'],3)
            self.assertEqual((report['output']['width'],report['output']['height']),(960,540))
            positions=iter([480.,505.,480.])
            def fake(frame,ctx):
                return frame,dict(two_accepted_boundaries=True,lane_rendered=True,
                    geometry=dict(lane_center_x_px=next(positions),reasons=[]),warnings=[],
                    initialization=[dict(method='primary'),dict(method='primary')])
            with patch('src.video_pipeline.process_frame',side_effect=fake):
                jumped=process_video(source,root/'synthetic_jumps')
            self.assertEqual(jumped['adjacent_center_change']['large_jumps'],2)
            self.assertEqual(jumped['adjacent_center_change']['over_30_px'],0)

    def test_nonempty_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.mp4';folder=root/'existing';folder.mkdir()
            sentinel=folder/'keep.txt';sentinel.write_text('preserve')
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(960,540))
            writer.write(np.zeros((540,960,3),np.uint8));writer.release()
            with self.assertRaises(FileExistsError):process_video(source,folder)
            self.assertEqual(sentinel.read_text(),'preserve');self.assertEqual(len(list(folder.iterdir())),1)

if __name__=='__main__':unittest.main()
