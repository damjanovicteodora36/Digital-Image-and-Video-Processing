import unittest
import numpy as np
from src.lane_detection import sliding_window_search


class FallbackTests(unittest.TestCase):
    def image(self):
        image=np.zeros((720,1280),np.uint8)
        image[:,310:320]=1
        image[0:320,950:960]=1
        return image

    def test_missing_lower_peak_and_bidirectional_tracking(self):
        image=self.image()
        old=sliding_window_search(image,enable_fallback=False)
        self.assertIsNone(old['seeds'][1])
        new=sliding_window_search(image)
        self.assertEqual(new['initialization'][1]['method'],'fallback')
        self.assertGreaterEqual(new['lanes']['right']['supported_windows'],4)
        self.assertEqual(new['lanes']['right']['count'],3200)
        self.assertEqual(new['lanes']['right']['y'].min(),0)
        self.assertEqual(new['lanes']['right']['y'].max(),319)

    def test_ambiguous_candidates_rejected(self):
        image=self.image(); image[:320,1050:1060]=1
        self.assertIsNone(sliding_window_search(image)['seeds'][1])

    def test_no_double_assignment(self):
        result=sliding_window_search(self.image(),margin=500)
        pairs=[set(zip(result['lanes'][s]['x'],result['lanes'][s]['y'])) for s in ('left','right')]
        self.assertFalse(pairs[0]&pairs[1])
        for side in ('left','right'):
            self.assertEqual(len(set(zip(result['lanes'][side]['x'],result['lanes'][side]['y']))),result['lanes'][side]['count'])

    def test_cross_band_ambiguity(self):
        image=self.image()
        image[:320,950:960]=0
        image[:160,950:960]=1
        image[160:320,1050:1060]=1
        result=sliding_window_search(image)
        self.assertIsNone(result['seeds'][1])
        self.assertIn('Competing',result['initialization'][1]['reason'])

    def test_seed_does_not_waive_support(self):
        image=self.image(); image[:240,950:960]=0
        result=sliding_window_search(image)
        self.assertEqual(result['initialization'][1]['method'],'fallback')
        self.assertFalse(result['passes_heuristics'])

    def test_primary_is_unchanged(self):
        image=self.image(); image[:,950:960]=1
        old,new=(sliding_window_search(image,enable_fallback=v) for v in (False,True))
        for side in ('left','right'):
            np.testing.assert_array_equal(old['lanes'][side]['x'],new['lanes'][side]['x'])


if __name__=='__main__': unittest.main()
