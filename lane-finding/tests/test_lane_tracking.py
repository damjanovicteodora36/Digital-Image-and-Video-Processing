"""Current-evidence, state transitions and disjoint temporal ownership."""
import unittest
from unittest.mock import patch
import numpy as np
from src.lane_tracking import LaneTracker, TrackingConfig, search_around_polynomials, process_frame

def mask(shift=0,right=True,rows=None):
    image=np.zeros((720,1280),np.uint8)
    yy=np.arange(720) if rows is None else np.asarray(rows)
    for base in ([320,960] if right else [320]):
        xx=np.rint(.00003*(yy-360)**2+base+shift).astype(int)
        for dx in range(-3,4):image[yy,xx+dx]=1
    return image

class TrackingTests(unittest.TestCase):
    def test_two_lanes_and_gradual_movement(self):
        tracker=LaneTracker()
        first=tracker.update(mask());self.assertEqual(first['state'],'INITIALIZATION')
        for delta in range(2,16,2):
            current=tracker.update(mask(delta));self.assertEqual(current['state'],'TRACKING')
            for side,lane in current['search']['lanes'].items():
                self.assertTrue(np.all(mask(delta)[lane['y'],lane['x']]==1))
                self.assertNotIn('supported_windows',lane)
                self.assertEqual(current['fits'][side]['supported_windows'],None)
                self.assertEqual(current['fits'][side]['supported_bands'],9)

    def test_empty_frame_never_reuses_old_pair(self):
        tracker=LaneTracker();tracker.update(mask())
        result=tracker.update(np.zeros((720,1280),np.uint8))
        self.assertEqual(result['state'],'LOST');self.assertIsNone(tracker.previous)
        self.assertIsNone(result['geometry']['offset_m']);self.assertFalse(result['geometry']['geometry_accepted'])
        self.assertEqual(len(result['attempts']),2)

    def test_missing_boundary(self):
        tracker=LaneTracker();tracker.update(mask())
        result=tracker.update(mask(right=False))
        self.assertFalse(result['accepted']);self.assertEqual(result['state'],'LOST')

    def test_sparse_and_concentrated_support(self):
        tracker=LaneTracker();tracker.update(mask())
        result=tracker.update(mask(rows=np.arange(100,160)))
        self.assertFalse(result['accepted'])
        self.assertTrue(any('support' in r.lower() or 'rows' in r.lower() for r in result['reasons']))

    def test_successful_reinitialization(self):
        tracker=LaneTracker();tracker.update(mask())
        empty=search_around_polynomials(np.zeros((720,1280),np.uint8),tracker.previous)
        with patch('src.lane_tracking.search_around_polynomials',return_value=empty):
            result=tracker.update(mask(2))
        self.assertTrue(result['accepted']);self.assertEqual(result['state'],'REINITIALIZATION')
        self.assertTrue(result['reinitialization_attempt'])

    def test_overlap_owned_once(self):
        binary=np.zeros((720,1280),np.uint8);binary[200:500,560:581]=1
        result=search_around_polynomials(binary,dict(left=[0,0,520],right=[0,0,620]))
        sets=[set(zip(lane['x'],lane['y'])) for lane in result['lanes'].values()]
        self.assertFalse(sets[0]&sets[1]);self.assertEqual(len(sets[0]|sets[1]),int(binary.sum()))

    def test_reset_between_videos(self):
        tracker=LaneTracker();tracker.update(mask());tracker.update(mask(2));tracker.reset()
        self.assertEqual(tracker.frame_index,0);self.assertIsNone(tracker.previous)
        self.assertEqual(tracker.update(mask())['state'],'INITIALIZATION')

    def test_abrupt_change_triggers_full_search_then_loss(self):
        tracker=LaneTracker();tracker.update(mask())
        result=tracker.update(mask(100))
        self.assertTrue(result['reinitialization_attempt']);self.assertEqual(result['state'],'LOST')
        self.assertTrue(any('abrupt' in reason for reason in result['reasons']))
        self.assertEqual(tracker.update(mask(100))['state'],'REINITIALIZATION')

    def test_process_frame_dimensions_and_no_evidence(self):
        image=np.zeros((720,1280,3),np.uint8)
        output,diagnostics=process_frame(image,LaneTracker())
        self.assertEqual(output.shape,image.shape);self.assertEqual(output.dtype,np.uint8)
        self.assertEqual(diagnostics['state'],'LOST');self.assertFalse(diagnostics['lane_rendered'])
        with self.assertRaises(ValueError):process_frame(image[:540,:960],LaneTracker())

    def test_configuration_validation(self):
        for options in [dict(margin=0),dict(min_supported_bands=10),dict(max_position_change_px=float('nan'))]:
            with self.assertRaises(ValueError):TrackingConfig(**options)

    def test_real_shape_change_uses_positions_not_coefficient_difference(self):
        tracker=LaneTracker();tracker.update(mask())
        image=np.zeros((720,1280),np.uint8);y=np.arange(720)
        for base in (320,960):
            x=np.rint(.00012*(y-360)**2+base).astype(int)
            for dx in range(-3,4):image[y,x+dx]=1
        result=tracker.update(image)
        self.assertEqual(result['state'],'TRACKING')
        self.assertTrue(result['accepted'])

if __name__=='__main__':unittest.main()
