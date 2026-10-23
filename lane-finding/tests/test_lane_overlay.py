import unittest
import numpy as np
import cv2
from src.lane_overlay import lane_polygon,inverse_warp,render_overlay
from src.perspective_transform import perspective_matrices,SOURCE_POINTS,DESTINATION_POINTS


class OverlayTests(unittest.TestCase):
    def fits(self):
        return {s:dict(reliable=True,coefficients=[0,0,x]) for s,x in [('left',20),('right',80)]}

    def test_polygon(self):
        polygon=lane_polygon(self.fits(),(100,100))
        self.assertEqual(polygon.shape,(200,2))
        self.assertEqual(polygon.dtype,np.int32)
        mask=np.zeros((100,100),np.uint8); cv2.fillPoly(mask,[polygon],1)
        self.assertEqual(mask.sum(),6100)

    def test_inverse_mapping(self):
        m,inv=perspective_matrices()
        points=np.array(DESTINATION_POINTS,np.float32)[None]
        np.testing.assert_allclose(cv2.perspectiveTransform(points,inv)[0],SOURCE_POINTS,atol=1e-3)
        image=np.zeros((100,100,3),np.uint8); image[20:80,20:80]=255
        np.testing.assert_array_equal(inverse_warp(image,np.eye(3),(100,100)),image)

    def test_unreliable_no_area(self):
        fits=self.fits(); fits['right']=dict(reliable=False,coefficients=None)
        self.assertIsNone(lane_polygon(fits,(300,300)))
        image=np.full((300,300,3),70,np.uint8)
        output,rendered=render_overlay(image,fits,dict(geometry_accepted=False,reasons=[]),np.eye(3))
        self.assertFalse(rendered)
        np.testing.assert_array_equal(output[160:],image[160:])

    def test_crossing_rejected(self):
        fits=self.fits(); fits['right']['coefficients']=[0,0,10]
        with self.assertRaises(ValueError): lane_polygon(fits,(100,100))


if __name__=='__main__': unittest.main()
