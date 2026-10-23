"""Synthetic regression tests for independent lane-fit gating."""

import unittest
import numpy as np
from src.lane_detection import fit_lane_polynomials


class PolynomialFittingTests(unittest.TestCase):
    def make_result(self, rows=None):
        y = np.arange(100, 600) if rows is None else rows
        coefficients = [0.0001, -0.05, 320]
        lanes = {side: dict(x=np.polyval(coefficients, y)+offset, y=y,
                            count=len(y), supported_windows=6)
                 for side, offset in (("left", 0), ("right", 600))}
        return dict(lanes=lanes, seeds=[320, 920], reasons=[])

    def test_known_coefficients_and_support(self):
        fits = fit_lane_polynomials(self.make_result(), (720, 1280))
        np.testing.assert_allclose(fits["left"]["coefficients"], [0.0001, -0.05, 320], atol=1e-9)
        np.testing.assert_allclose(fits["right"]["coefficients"], [0.0001, -0.05, 920], atol=1e-9)
        self.assertEqual(fits["left"]["supported_rows"].sum(), 500)
        self.assertEqual(fits["left"]["extrapolated_rows"].sum(), 220)

    def test_insufficient_pixels_and_rows(self):
        for rows in (np.array([], dtype=int), np.arange(20), np.repeat([100, 101], 200)):
            self.assertFalse(fit_lane_polynomials(self.make_result(rows), (720, 1280))["left"]["reliable"])

    def test_unreliable_right_does_not_reject_left(self):
        result = self.make_result()
        result["lanes"]["right"]["supported_windows"] = 3
        result["reasons"] = ["right: insufficient window support"]
        fits = fit_lane_polynomials(result, (720, 1280))
        self.assertTrue(fits["left"]["reliable"])
        self.assertIsNone(fits["right"]["coefficients"])

    def test_internal_gaps_are_not_extrapolation(self):
        rows = np.concatenate([np.arange(100, 300), np.arange(400, 600)])
        fit = fit_lane_polynomials(self.make_result(rows), (720, 1280))["left"]
        self.assertTrue(fit["interpolated_rows"][350])
        self.assertFalse(fit["extrapolated_rows"][350])

    def test_obvious_invalid_curve(self):
        result = self.make_result()
        result["lanes"]["left"]["x"] += 2000
        self.assertFalse(fit_lane_polynomials(result, (720, 1280))["left"]["reliable"])


if __name__ == "__main__":
    unittest.main()
