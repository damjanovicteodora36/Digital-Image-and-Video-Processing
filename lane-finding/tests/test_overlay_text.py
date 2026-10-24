import copy
import unittest
import numpy as np
from src.lane_overlay import format_overlay_text, render_overlay

class OverlayTextTests(unittest.TestCase):
    def data(self):
        fits={s:dict(reliable=True, coefficients=[0,0,x]) for s,x in [('left',320),('right',960)]}
        geometry=dict(geometry_accepted=True,offset_m=-.21,bottom_support={},reasons=['retained warning'],
            curvature={s:dict(radius_m=2144,status='estimated') for s in fits})
        return fits,geometry

    def test_radius_format_and_display_threshold(self):
        f,g=self.data()
        self.assertIn('Left curvature: ~2,144 m',format_overlay_text(f,g))
        for radius in (10000,1e9,float('inf')):
            g['curvature']['right']['radius_m']=radius
            self.assertIn('Right curvature: Nearly straight',format_overlay_text(f,g))
        g['curvature']['right']['radius_m']=15000
        self.assertIn('Right curvature: ~15,000 m',format_overlay_text(f,g,nearly_straight_m=20000))

    def test_straight_status_with_serialized_null(self):
        f,g=self.data();g['curvature']['right']=dict(radius_m=None,status='practically_straight')
        self.assertIn('Right curvature: Nearly straight',format_overlay_text(f,g))

    def test_offset_sign_and_center_threshold(self):
        f,g=self.data()
        for offset,label in [(-.21,'0.21 m left of center'),(.21,'0.21 m right of center'),(0,'Centered'),(.05,'Centered'),(-.05,'Centered'),(.051,'0.05 m right of center')]:
            g['offset_m']=offset
            self.assertIn('Vehicle position: '+label,format_overlay_text(f,g))
        g['offset_m']=.04
        self.assertIn('Vehicle position: 0.04 m right of center',format_overlay_text(f,g,centered_m=.01))

    def test_extrapolation_and_nonmutation(self):
        f,g=self.data();g['bottom_support']={'right':'extrapolated'};before=copy.deepcopy(g)
        self.assertEqual(format_overlay_text(f,g)[-1],'Approximate estimates | Right boundary extrapolated')
        self.assertEqual(g,before)
        g['bottom_support']['left']='extrapolated'
        self.assertIn('Both boundaries extrapolated',format_overlay_text(f,g)[-1])

    def test_unavailable_and_rejected_geometry(self):
        for bad_fit in (False,True):
            f,g=self.data()
            if bad_fit:f['right']['reliable']=False
            else:g['geometry_accepted']=False
            self.assertEqual(format_overlay_text(f,g),['LANE DETECTION','Lane unavailable','Insufficient reliable lane boundaries'])
            image=np.full((540,960,3),70,np.uint8)
            output,accepted=render_overlay(image,f,g,np.eye(3))
            self.assertFalse(accepted)
            np.testing.assert_array_equal(output[200:],image[200:])

    def test_layout_both_resolutions_and_road_unchanged(self):
        for width,height in ((960,540),(1280,720)):
            f,g=self.data()
            for side,x in [('left',width*.25),('right',width*.75)]:f[side]['coefficients']=[0,0,x]
            image=np.full((height,width,3),70,np.uint8)
            g['bottom_support']={'left':'extrapolated','right':'extrapolated'}
            output,accepted=render_overlay(image,f,g,np.eye(3))
            self.assertTrue(accepted);self.assertEqual(output.shape,image.shape)
            # Panel stays above the horizon; the green area remains visible below it.
            self.assertTrue(np.all(output[int(height*.28),10]==70))
            self.assertGreater(output[height-1,width//2,1],70)

    def test_invalid_display_parameters(self):
        f,g=self.data()
        for kwargs in ({'nearly_straight_m':0},{'nearly_straight_m':float('inf')},{'centered_m':-.1}):
            with self.assertRaises(ValueError):format_overlay_text(f,g,**kwargs)
