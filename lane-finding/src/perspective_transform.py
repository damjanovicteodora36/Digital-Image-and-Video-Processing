"""Candidate bird's-eye transform for manual inspection, not lane fitting."""

import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .thresholding import binarize_image, load_calibration, undistort_image, save_image
    from .thresholding import threshold_image, create_roi_mask
else:
    from thresholding import binarize_image, load_calibration, undistort_image, save_image
    from thresholding import threshold_image, create_roi_mask


PROJECT_DIR = Path(__file__).resolve().parents[1]
# Pixel candidates selected on the undistorted 1280x720 straight_lines1 image.
# Order: top-left, top-right, bottom-right, bottom-left.
SOURCE_POINTS = ((540, 490), (748, 490), (1042, 680), (263, 680))
DESTINATION_POINTS = ((320, 0), (960, 0), (960, 719), (320, 719))


def validate_points(points, name):
    """Require a convex TL, TR, BR, BL quadrilateral in image coordinates."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name}: expected four finite (x, y) points")
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError(f"{name}: points must be distinct")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if (not np.all(cross > 1e-3) or points[0, 0] >= points[1, 0]
            or points[3, 0] >= points[2, 0]
            or max(points[:2, 1]) >= min(points[2:, 1])):
        raise ValueError(f"{name}: require convex TL, TR, BR, BL order")
    return points


def perspective_matrices(source_points=SOURCE_POINTS, destination_points=DESTINATION_POINTS):
    """Return validated forward and reverse homographies for pixel coordinates."""
    source = validate_points(source_points, "source")
    destination = validate_points(destination_points, "destination")
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    if (not np.isfinite(matrix).all() or not np.isfinite(inverse).all()
            or np.linalg.matrix_rank(matrix) < 3):
        raise ValueError("Degenerate perspective transform")
    mapped = cv2.perspectiveTransform(source[None], matrix)[0]
    if not np.allclose(mapped, destination, atol=1e-3):
        raise ValueError("Unstable perspective transform")
    return matrix, inverse


def warp_binary(mask, matrix, output_size=None):
    """Warp a uint8 0/1 mask using nearest-neighbor interpolation."""
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("Expected a two-dimensional uint8 0/1 mask")
    size = mask.shape[1::-1] if output_size is None else tuple(output_size)
    return cv2.warpPerspective(mask, matrix, size, flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def mark_points(image, points):
    """Annotate a copy; never draw on a numerical mask."""
    marked = image.copy()
    pixels = np.rint(points).astype(np.int32)
    cv2.polylines(marked, [pixels], True, (0, 255, 0), 2)
    for index, (x, y) in enumerate(pixels):
        cv2.circle(marked, (int(x), int(y)), 7, (0, 0, 255), -1)
        cv2.putText(marked, str(index), (int(x)+10, max(20, int(y)-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return marked


def generate_candidates(source_points=SOURCE_POINTS, destination_points=DESTINATION_POINTS,
                        output_dir=PROJECT_DIR / "results/perspective_transform"):
    """Generate three candidate comparisons using one fixed camera geometry."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration()
    matrix, inverse = perspective_matrices(source_points, destination_points)
    records = []
    for name in ("straight_lines1", "test1", "test3"):
        image = cv2.imread(str(PROJECT_DIR / "test_images" / f"{name}.jpg"))
        if image is None:
            raise OSError(f"Could not read {name}")
        if image.shape[1::-1] != (1280, 720):
            raise ValueError("Default point candidates require 1280x720; select new points for other sizes")
        corrected = image.shape[1::-1] == calibration[2]
        prepared = undistort_image(image, calibration) if corrected else image
        binary = binarize_image(prepared)
        warped = warp_binary(binary, matrix)
        before = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        after = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2BGR)
        source_view = mark_points(prepared, source_points)
        destination_view = mark_points(after, destination_points)
        panels = [prepared.copy(), before, destination_view, source_view]
        labels = ["Undistorted" if corrected else "Original - no correction",
                  "Binary before warp", "Bird's-eye candidate + destination points",
                  "Source points: TL, TR, BR, BL"]
        for panel, label in zip(panels, labels):
            cv2.rectangle(panel, (0, 25), (panel.shape[1]-1, 65), (0, 0, 0), -1)
            cv2.putText(panel, label, (12, 53), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 1, cv2.LINE_AA)
        save_image(output_dir / f"{name}_comparison.jpg",
                   np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
        save_image(output_dir / f"{name}_source_points.jpg", source_view)
        save_image(output_dir / f"{name}_destination_points.jpg", destination_view)
        save_image(output_dir / f"{name}_binary_before.png", binary * 255)
        save_image(output_dir / f"{name}_binary_warped.png", warped * 255)
        np.savez_compressed(output_dir / f"{name}_masks.npz", before=binary, warped=warped)
        records.append(dict(image=f"{name}.jpg", image_size=list(image.shape[1::-1]),
                            distortion_correction_applied=corrected,
                            comparison=f"{name}_comparison.jpg"))
    np.savez(output_dir / "perspective_parameters.npz", M=matrix, Minv=inverse,
             source_points=source_points, destination_points=destination_points, image_size=(1280, 720))
    report = dict(source_points=source_points, destination_points=destination_points,
                  point_order="TL, TR, BR, BL", M=matrix.tolist(), Minv=inverse.tolist(),
                  candidate_basis="Lane marking candidates inspected on undistorted straight_lines1, not ROI boundaries",
                  status="paint_aligned accepted after visual review; not a metric ground-plane calibration", records=records)
    (output_dir / "perspective_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def generate_geometry_experiment(output_dir=PROJECT_DIR / "results/perspective_transform/experiments"):
    """Compare ROI effects and two point sets without changing defaults."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(PROJECT_DIR / "test_images/straight_lines1.jpg"))
    if image is None or image.shape[1::-1] != (1280, 720):
        raise ValueError("Expected readable 1280x720 straight_lines1")
    prepared = undistort_image(image, load_calibration())
    with_roi = binarize_image(prepared)
    without_roi = threshold_image(prepared)[2]
    roi = create_roi_mask(image.shape)
    # Both upper points now lie on visible paint at the same scanline.
    alternative = ((540, 490), (748, 490), (1042, 680), (263, 680))
    variants = {"baseline": ((575, 460), (698, 460), (1050, 680), (260, 680)),
                "paint_aligned": alternative}
    records, rows = [], []

    def label(panel, text):
        panel = panel.copy()
        cv2.rectangle(panel, (0, 28), (1279, 70), (0, 0, 0), -1)
        cv2.putText(panel, text, (10, 57), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    for name, points in variants.items():
        matrix, inverse = perspective_matrices(points)
        trapezoid = np.zeros(with_roi.shape, np.uint8)
        cv2.fillPoly(trapezoid, [np.array(points, np.int32)], 1)
        removed = without_roi & (1-roi) & trapezoid
        excluded_area = trapezoid & (1-roi)
        overlay = mark_points(prepared, points)
        overlay[removed == 1] = (0, 0, 255)
        warped_roi = warp_binary(with_roi, matrix)
        warped_no_roi = warp_binary(without_roi, matrix)
        color_warp = cv2.warpPerspective(prepared, matrix, (1280, 720), flags=cv2.INTER_LINEAR)
        panels = [label(overlay, f"{name}: source points; removed pixels red"),
                  label(mark_points(color_warp, DESTINATION_POINTS), f"{name}: color warp"),
                  label(cv2.cvtColor(warped_roi*255, cv2.COLOR_GRAY2BGR), "Binary warp WITH ROI"),
                  label(cv2.cvtColor(warped_no_roi*255, cv2.COLOR_GRAY2BGR), "Binary warp WITHOUT ROI")]
        comparison = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        save_image(output_dir / f"straight_lines1_{name}_comparison.jpg", comparison)
        save_image(output_dir / f"straight_lines1_{name}_color_warp.jpg", color_warp)
        save_image(output_dir / f"straight_lines1_{name}_removed_inside.png", removed*255)
        np.savez_compressed(output_dir / f"straight_lines1_{name}_masks.npz",
                            with_roi=with_roi, without_roi=without_roi, removed=removed,
                            warped_roi=warped_roi, warped_no_roi=warped_no_roi)
        rows.append(np.hstack(panels))
        records.append(dict(variant=name, source_points=points, destination_points=DESTINATION_POINTS,
                            M=matrix.tolist(), Minv=inverse.tolist(),
                            trapezoid_pixels_outside_roi=int(excluded_area.sum()),
                            selected_pixels_removed_inside_trapezoid=int(removed.sum())))
    save_image(output_dir / "straight_lines1_variants_comparison.jpg", np.vstack(rows))
    report = dict(image="straight_lines1.jpg", distortion_correction_applied=True,
                  alternative_basis="Centers of visible HLS paint runs: y=490 left x=534..546, right x=745..750; y=680 left x=248..278, right x=1031..1053. Rounded centers selected manually; no lane fitting.",
                  caveat="Removed selected pixels are not semantic lane labels. Geometry remains a candidate.",
                  variants=records)
    (output_dir / "geometry_experiment_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="store_true", help="Compare ROI and point candidates")
    args = parser.parse_args()
    if args.experiment:
        generate_geometry_experiment()
    else:
        generate_candidates()
