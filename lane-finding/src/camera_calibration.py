"""Detect and draw chessboard corners in the calibration photographs."""

from pathlib import Path
from collections import Counter
import json

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATTERN_SIZE = (9, 6)  # Inner corners: columns, rows (10 x 7 squares).


def find_corners(image, pattern_size=PATTERN_SIZE):
    """Return OpenCV's detection status and corner coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def detect_calibration_corners(
    image_dir=PROJECT_DIR / "camera_cal",
    output_dir=PROJECT_DIR / "results" / "calibration",
    pattern_size=PATTERN_SIZE,
):
    """Process all calibration JPGs and return success/failure path lists."""
    image_dir, output_dir = Path(image_dir), Path(output_dir)
    image_paths = sorted(
        image_dir.glob("calibration*.jpg"),
        key=lambda path: int(path.stem.removeprefix("calibration")),
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration photographs found in {image_dir}")
    if image_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the input directory")

    output_dir.mkdir(parents=True, exist_ok=True)
    successful, failed = [], []
    print(f"Images: {len(image_paths)}; inner corners: {pattern_size}")
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path.name}: FAILED (could not read image)")
            failed.append(image_path)
            continue

        found, corners = find_corners(image, pattern_size)
        if not found:
            print(f"{image_path.name}: FAILED (complete pattern not found)")
            failed.append(image_path)
            continue

        # Draw on a copy; source photographs are never overwritten.
        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, pattern_size, corners, found)
        output_path = output_dir / f"{image_path.stem}_corners.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save annotated image: {output_path}")
        successful.append(image_path)
        print(f"{image_path.name}: OK ({len(corners)} corners) -> {output_path.name}")

    print(f"Detected: {len(successful)}/{len(image_paths)}; failed: {len(failed)}")
    if failed:
        print("Failed images: " + ", ".join(path.name for path in failed))
    return successful, failed


def calibrate_camera(image_paths, output_dir=PROJECT_DIR / "results" / "calibration"):
    """Calibrate at the most common resolution, excluding other image sizes."""
    if len(image_paths) < 3:
        raise ValueError("At least three successful photographs are required")
    images = [(Path(path), cv2.imread(str(path))) for path in image_paths]
    if any(image is None for _, image in images):
        raise OSError("Could not read a calibration photograph")
    sizes = Counter(image.shape[1::-1] for _, image in images)
    image_size = sizes.most_common(1)[0][0]
    print(f"Calibration resolutions (width, height): {dict(sizes)}")

    # Coordinates use one square as the unit; the board lies on Z = 0.
    grid = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    object_points, image_points, names, excluded = [], [], [], []
    for path, image in images:
        if image.shape[1::-1] != image_size:
            print(f"Excluded from calibration: {path.name} (different resolution)")
            excluded.append(path.name)
            continue
        found, corners = find_corners(image)
        if not found:
            raise ValueError(f"Previously detected pattern missing: {path.name}")
        object_points.append(grid.copy())
        image_points.append(corners)
        names.append(path.name)
    if len(names) < 3:
        raise ValueError("Insufficient photographs at the common resolution")

    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_image_mean, per_image_rms = [], []
    for name, obj, observed, rotation, translation in zip(
        names, object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        distances = np.linalg.norm(observed.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        per_image_mean.append(float(distances.mean()))
        per_image_rms.append(float(np.sqrt(np.mean(distances ** 2))))
        print(f"{name}: mean error {per_image_mean[-1]:.6f} px; RMS {per_image_rms[-1]:.6f} px")

    mean_error = float(np.mean(per_image_mean))  # Every photograph has 54 corners.
    print(f"Used photographs: {len(names)}; mean error: {mean_error:.6f} px; RMS: {rms:.6f} px")
    print("Camera matrix:\n", matrix)
    print("Distortion coefficients (k1, k2, p1, p2, k3):\n", distortion)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "camera_parameters.npz", camera_matrix=matrix,
        distortion_coefficients=distortion, image_size=np.array(image_size),
        pattern_size=np.array(PATTERN_SIZE), square_size=1.0,
        rms=rms, mean_reprojection_error=mean_error,
        per_image_mean_error=per_image_mean, per_image_rms=per_image_rms,
        image_names=np.array(names), rvecs=np.array(rvecs), tvecs=np.array(tvecs),
    )
    report = dict(
        image_size=list(image_size), pattern_size=list(PATTERN_SIZE),
        used_images=names, excluded_resolution=excluded,
        camera_matrix=matrix.tolist(), distortion_coefficients=distortion.tolist(),
        rms_pixels=rms, mean_error_pixels=mean_error,
        per_image=[dict(image=name, mean_pixels=mean, rms_pixels=error)
                   for name, mean, error in zip(names, per_image_mean, per_image_rms)],
        resolution_policy="Use only the most common resolution; no assumed resize/crop mapping.",
    )
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return matrix, distortion, image_size


def save_undistortion_comparison(image_path, parameters_path, output_path, *, resized=False):
    """Load saved parameters; allow scaling only for known full-frame resizing."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read {image_path}")
    with np.load(parameters_path, allow_pickle=False) as parameters:
        matrix = parameters["camera_matrix"].copy()
        distortion = parameters["distortion_coefficients"]
        calibrated_size = tuple(parameters["image_size"])
    width, height = image.shape[1::-1]
    if (width, height) != calibrated_size:
        if not resized:
            raise ValueError("Resolution differs: confirm full-frame resizing with resized=True")
        # A full-frame resize scales fx/cx and fy/cy; distortion stays unchanged.
        matrix = np.diag([width / calibrated_size[0], height / calibrated_size[1], 1.0]) @ matrix
    corrected = cv2.undistort(image, matrix, distortion, None, matrix)
    comparison = np.hstack([image, corrected])
    for label, x in [("Original", 20), ("Undistorted", width + 20)]:
        cv2.putText(comparison, label, (x, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(comparison, label, (x, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Could not write {output_path}")
    return comparison


def main():
    """Detect corners, calibrate, and demonstrate saved parameter reuse."""
    successful, _ = detect_calibration_corners()
    output_dir = PROJECT_DIR / "results" / "calibration"
    calibrate_camera(successful, output_dir)
    for source, name in [
        (PROJECT_DIR / "camera_cal" / "calibration2.jpg", "calibration2_comparison.jpg"),
        (PROJECT_DIR / "test_images" / "test1.jpg", "test1_comparison.jpg"),
    ]:
        save_undistortion_comparison(source, output_dir / "camera_parameters.npz", output_dir / name)


if __name__ == "__main__":
    main()
