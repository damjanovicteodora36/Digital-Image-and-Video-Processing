"""Detect and draw chessboard corners in the calibration photographs."""

from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATTERN_SIZE = (9, 6)  # Inner corners: columns, rows (10 x 7 squares).


def find_corners(image, pattern_size=PATTERN_SIZE):
    """Return OpenCV's detection status and corner coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.findChessboardCorners(gray, pattern_size)


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


if __name__ == "__main__":
    detect_calibration_corners()
