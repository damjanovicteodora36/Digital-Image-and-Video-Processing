import unittest
import numpy as np
from src.lane_geometry import curvature_radius, compute_lane_geometry, uncertainty_labels


def fit(coefficients, reliable=True):
    return dict(coefficients=coefficients, reliable=reliable, reasons=[])


class LaneGeometryTests(unittest.TestCase):
    def test_uncertainty_labels_do_not_change_acceptance(self):
        geometry = dict(fit_reliable=dict(left=True, right=True),
                        bottom_support=dict(left="observed", right="extrapolated"),
                        curvature=dict(left=dict(radius_m=100, status="estimated"),
                                       right=dict(radius_m=500, status="estimated")), geometry_accepted=True)
        labels = uncertainty_labels(geometry)
        self.assertTrue(any("EXTRAPOLATED" in s for s in labels))
        self.assertTrue(any("radii differ" in s for s in labels))
        self.assertTrue(geometry["geometry_accepted"])
        self.assertFalse(any("radii differ" in s for s in uncertainty_labels(geometry, 6)))
        geometry["fit_reliable"]["right"] = False
        geometry["curvature"]["right"] = dict(radius_m=None, status="unreliable_fit")
        self.assertTrue(any("UNRELIABLE" in s for s in uncertainty_labels(geometry)))
        self.assertFalse(any("radii differ" in s for s in uncertainty_labels(geometry)))

    def test_straight(self):
        self.assertTrue(np.isinf(curvature_radius(fit([0, 0, 320]), 720)["radius_m"]))

    def test_known_curvature(self):
        # x_m = 0.01*y_m**2; curvature at y_m=10 is analytic.
        sx, sy = 0.01, 0.1
        result = curvature_radius(fit([0.01*sy**2/sx, 0, 300]), 101, sy, sx)
        self.assertAlmostEqual(result["radius_m"], (1+0.2**2)**1.5/0.02)

    def test_offset_signs(self):
        for shift in (-40, 40):
            lanes = dict(left=fit([0, 0, 320+shift]), right=fit([0, 0, 960+shift]))
            self.assertAlmostEqual(compute_lane_geometry(lanes, (720, 1280))["offset_m"], -shift*3.7/640)

    def test_crossed(self):
        lanes = dict(left=fit([0, 0, 320]), right=fit([0, -1, 960]))
        self.assertIsNone(compute_lane_geometry(lanes, (720, 1280))["offset_m"])

    def test_unreliable(self):
        lanes = dict(left=fit([0, 0, 320]), right=fit(None, False))
        result = compute_lane_geometry(lanes, (720, 1280))
        self.assertIsNone(result["offset_m"])
        self.assertIsNone(result["curvature"]["right"]["radius_m"])

    def test_width_and_variation(self):
        for coeff in ([0, 0, 400], [0, 0.5, 600]):
            result = compute_lane_geometry(dict(left=fit([0, 0, 320]), right=fit(coeff)), (720, 1280))
            self.assertIsNone(result["offset_m"])


if __name__ == "__main__":
    unittest.main()
