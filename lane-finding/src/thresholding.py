"""Extract color and gradient candidates using saved camera calibration."""

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = dict(
    white_lightness=(200, 255), yellow_hue=(15, 40),
    yellow_saturation=(100, 255), yellow_lightness=(40, 255),
    sobel_threshold=(40, 100), sobel_kernel=3,
)
COMPARISON_IMAGES = {"test1", "test3", "straight_lines1"}
DEFAULT_ROI = ((0.0, 1.0), (0.38, 0.58), (0.68, 0.58), (1.0, 1.0))
SOBEL_RANGES = ((20, 100), (40, 100), (60, 150))


def load_calibration(path=PROJECT_DIR / "results/calibration/camera_parameters.npz"):
    """Load existing parameters without running calibration."""
    with np.load(path, allow_pickle=False) as data:
        return (data["camera_matrix"].copy(), data["distortion_coefficients"].copy(),
                tuple(int(value) for value in data["image_size"]))


def undistort_image(image, calibration, *, resized=False):
    """Require matching resolution unless full-frame resizing is confirmed."""
    matrix, distortion, size = calibration
    width, height = image.shape[1::-1]
    if (width, height) != size:
        if not resized:
            raise ValueError(f"Resolution {width}x{height} differs from calibration {size}; "
                             "same-camera full-frame resizing is not confirmed")
        matrix = np.diag([width / size[0], height / size[1], 1.0]) @ matrix
    return cv2.undistort(image, matrix, distortion, None, matrix)


def in_range(channel, bounds):
    """Return an inclusive threshold selection."""
    return (channel >= bounds[0]) & (channel <= bounds[1])


def color_mask(hls, white_lightness=(200, 255), yellow_hue=(15, 40),
               yellow_saturation=(100, 255), yellow_lightness=(40, 255)):
    """Select bright white and sufficiently saturated yellow pixels."""
    hue, lightness, saturation = cv2.split(hls)
    white = in_range(lightness, white_lightness)
    yellow = (in_range(hue, yellow_hue) & in_range(saturation, yellow_saturation)
              & in_range(lightness, yellow_lightness))
    return (white | yellow).astype(np.uint8)


def sobel_x_mask(lightness, threshold=(40, 100), kernel_size=3):
    """Threshold the absolute X derivative scaled to 0..255."""
    if kernel_size not in (1, 3, 5, 7):
        raise ValueError("Sobel kernel must be 1, 3, 5, or 7")
    gradient = np.abs(cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=kernel_size))
    maximum = float(gradient.max())
    if maximum == 0:
        return np.zeros(lightness.shape, dtype=np.uint8)
    scaled = (255 * gradient / maximum).astype(np.uint8)
    return in_range(scaled, threshold).astype(np.uint8)


def threshold_image(image, white_lightness=(200, 255), yellow_hue=(15, 40),
                    yellow_saturation=(100, 255), yellow_lightness=(40, 255),
                    sobel_threshold=(40, 100), sobel_kernel=3):
    """Return color, gradient, and combined 0/1 masks for a BGR image."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    colors = color_mask(hls, white_lightness, yellow_hue, yellow_saturation, yellow_lightness)
    gradient = sobel_x_mask(hls[:, :, 1], sobel_threshold, sobel_kernel)
    return colors, gradient, (colors | gradient)


def save_image(path, image):
    """Raise a clear error when OpenCV cannot write an output."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def save_comparison(path, image, masks):
    """Build a labeled 2 by 2 comparison without labeling the actual masks."""
    panels = [image.copy()] + [cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR) for mask in masks]
    for panel, label in zip(panels, ["Undistorted", "Color", "Sobel X", "Combined (OR)"]):
        cv2.rectangle(panel, (0, 0), (300, 45), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)
    save_image(path, np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))


def process_test_images(image_dir=PROJECT_DIR / "test_images",
                        output_dir=PROJECT_DIR / "results/thresholding", *,
                        parameters=None, resized=False):
    """Process readable, compatible images and record successes and failures."""
    calibration = load_calibration()
    settings = DEFAULT_PARAMETERS.copy()
    if parameters:
        settings.update(parameters)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    paths = sorted(p for p in Path(image_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in paths:
        record = dict(image=path.name)
        image = cv2.imread(str(path))
        try:
            if image is None:
                raise OSError("Could not read image")
            record["image_size"] = list(image.shape[1::-1])
            corrected = undistort_image(image, calibration, resized=resized)
            masks = threshold_image(corrected, **settings)
            outputs = []
            for label, mask in zip(["color", "gradient", "combined"], masks):
                target = output_dir / f"{path.stem}_{label}.png"
                # PNG previews use 0/255; NumPy masks retain exact 0/1 values.
                save_image(target, mask * 255)
                outputs.append(str(target.relative_to(output_dir)))
            target = output_dir / f"{path.stem}_undistorted.jpg"
            save_image(target, corrected)
            outputs.append(target.name)
            target = output_dir / f"{path.stem}_masks.npz"
            np.savez_compressed(target, color=masks[0], gradient=masks[1], combined=masks[2])
            outputs.append(target.name)
            if path.stem in COMPARISON_IMAGES:
                target = output_dir / f"{path.stem}_comparison.jpg"
                save_comparison(target, corrected, masks)
                outputs.append(target.name)
            record.update(status="processed", outputs=outputs,
                          selected_fraction={label: float(mask.mean()) for label, mask in
                                             zip(["color", "gradient", "combined"], masks)})
            print(f"{path.name}: OK ({image.shape[1]}x{image.shape[0]})")
        except (ValueError, OSError, cv2.error) as error:
            record.update(status="failed_or_skipped", reason=str(error))
            print(f"{path.name}: FAILED/SKIPPED: {error}")
        records.append(record)
    processed = sum(row["status"] == "processed" for row in records)
    report = dict(total=len(paths), processed=processed, failed_or_skipped=len(paths)-processed,
                  parameters=settings, combination="color OR gradient", records=records,
                  resolution_policy="Exact calibration resolution unless same-camera full-frame resize is explicitly confirmed.")
    (output_dir / "thresholding_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Processed: {processed}/{len(paths)}; failed/skipped: {len(paths)-processed}")
    return report


def create_roi_mask(image_shape, vertices=DEFAULT_ROI):
    """Rasterize normalized (x, y) polygon vertices into a 0/1 mask."""
    height, width = image_shape[:2]
    points = np.asarray(vertices, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or np.any((points < 0) | (points > 1))):
        raise ValueError("ROI needs at least three finite vertices in [0, 1]")
    pixels = np.rint(points * [width - 1, height - 1]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    return mask


def save_roi_comparison(path, image, masks, roi, corrected, threshold):
    """Show source, HLS, Sobel, OR, ROI result, and polygon location."""
    source_label = "Undistorted" if corrected else "Original - no correction"
    outline = image.copy()
    points = np.rint(np.asarray(roi) * [image.shape[1]-1, image.shape[0]-1]).astype(np.int32)
    cv2.polylines(outline, [points], True, (0, 255, 0), 3)
    panels = [image.copy()] + [cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR) for m in masks] + [outline]
    labels = [source_label, "HLS", f"Sobel X {threshold[0]}-{threshold[1]}",
              "HLS OR Sobel - before ROI", "HLS OR Sobel - after ROI", "ROI outline"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, 42), (0, 0, 0), -1)
        cv2.putText(panel, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
    save_image(path, np.vstack([np.hstack(panels[i:i+2]) for i in (0, 2, 4)]))


def process_roi_experiments(image_dir=PROJECT_DIR / "test_images",
                            output_dir=PROJECT_DIR / "results/thresholding/roi_experiments",
                            *, vertices=DEFAULT_ROI, sobel_ranges=SOBEL_RANGES,
                            parameters=None):
    """Compare thresholds without selecting a winner or overwriting v1 outputs."""
    calibration = load_calibration()
    settings = DEFAULT_PARAMETERS.copy()
    if parameters:
        settings.update(parameters)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    paths = sorted(p for p in Path(image_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in paths:
        record = dict(image=path.name)
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError("Could not read image")
            corrected = image.shape[1::-1] == calibration[2]
            prepared = undistort_image(image, calibration) if corrected else image.copy()
            record.update(image_size=list(image.shape[1::-1]), distortion_correction_applied=corrected,
                          correction_note=("Saved calibration applied at matching resolution" if corrected else
                                           "Not applied: same-camera resize/crop mapping is unconfirmed"))
            roi = create_roi_mask(image.shape, vertices)
            outputs, experiments = [], []
            for bounds in sobel_ranges:
                variant = settings.copy()
                variant["sobel_threshold"] = bounds
                colors, gradient, combined = threshold_image(prepared, **variant)
                after_roi = combined & roi
                masks = dict(color=colors, gradient=gradient, combined_before_roi=combined,
                             combined_after_roi=after_roi, roi=roi)
                prefix = f"{path.stem}_sobel_{bounds[0]}_{bounds[1]}"
                for label, mask in masks.items():
                    target = output_dir / f"{prefix}_{label}.png"
                    save_image(target, mask * 255)
                    outputs.append(target.name)
                target = output_dir / f"{prefix}_masks.npz"
                np.savez_compressed(target, **masks)
                outputs.append(target.name)
                if path.stem in COMPARISON_IMAGES:
                    target = output_dir / f"{prefix}_comparison.jpg"
                    save_roi_comparison(target, prepared, [colors, gradient, combined, after_roi],
                                        vertices, corrected, bounds)
                    outputs.append(target.name)
                experiments.append(dict(sobel_threshold=list(bounds),
                                        selected_fraction={k: float(m.mean()) for k, m in masks.items()}))
            record.update(status="processed", outputs=outputs, experiments=experiments)
            print(f"{path.name}: OK; distortion correction: {corrected}")
        except (OSError, ValueError, cv2.error) as error:
            record.update(status="failed", reason=str(error))
            print(f"{path.name}: FAILED: {error}")
        records.append(record)
    processed = sum(r["status"] == "processed" for r in records)
    report = dict(total=len(paths), processed=processed, failed=len(paths)-processed,
                  roi_vertices=vertices, parameters=settings, sobel_ranges=sobel_ranges,
                  selection_note="No threshold selected automatically; pixel fractions are not accuracy metrics.",
                  records=records)
    (output_dir / "roi_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Processed: {processed}/{len(paths)}; failed: {len(paths)-processed}")
    return report


def binarize_image(image, *, vertices=DEFAULT_ROI, white_lightness=(200, 255),
                   yellow_hue=(15, 40), yellow_saturation=(100, 255),
                   yellow_lightness=(40, 255), sobel_threshold=(40, 100), sobel_kernel=3):
    """Return a uint8 0/1 HLS OR Sobel mask restricted to the road ROI.

    Input is a nonempty uint8 BGR image at any resolution. Undistortion is
    the caller's responsibility; this function performs no file I/O.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("Expected a nonempty uint8 BGR image")
    _, _, combined = threshold_image(
        image, white_lightness, yellow_hue, yellow_saturation,
        yellow_lightness, sobel_threshold, sobel_kernel,
    )
    return combined & create_roi_mask(image.shape, vertices)


def process_final_images(image_dir=PROJECT_DIR / "test_images",
                         output_dir=PROJECT_DIR / "results/thresholding/final"):
    """Run the selected configuration while preserving experimental outputs."""
    calibration = load_calibration()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    paths = sorted(p for p in Path(image_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in paths:
        record = dict(image=path.name)
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError("Could not read image")
            corrected = image.shape[1::-1] == calibration[2]
            prepared = undistort_image(image, calibration) if corrected else image
            binary = binarize_image(prepared)
            target = output_dir / f"{path.stem}_binary.png"
            save_image(target, binary * 255)
            np.savez_compressed(output_dir / f"{path.stem}_binary.npz", binary=binary)
            outputs = [target.name, f"{path.stem}_binary.npz"]
            if path.stem in COMPARISON_IMAGES:
                colors, gradient, combined = threshold_image(prepared)
                target = output_dir / f"{path.stem}_comparison.jpg"
                save_roi_comparison(target, prepared, [colors, gradient, combined, binary],
                                    DEFAULT_ROI, corrected, DEFAULT_PARAMETERS["sobel_threshold"])
                outputs.append(target.name)
            record.update(status="processed", image_size=list(image.shape[1::-1]),
                          distortion_correction_applied=corrected,
                          correction_note=("Saved calibration applied" if corrected else
                                           "Not applied: same-camera resize/crop mapping is unconfirmed"),
                          outputs=outputs)
            print(f"{path.name}: OK; distortion correction: {corrected}")
        except (OSError, ValueError, cv2.error) as error:
            record.update(status="failed", reason=str(error))
            print(f"{path.name}: FAILED: {error}")
        records.append(record)
    processed = sum(r["status"] == "processed" for r in records)
    report = dict(total=len(paths), processed=processed, failed=len(paths)-processed,
                  parameters=DEFAULT_PARAMETERS, roi_vertices=DEFAULT_ROI,
                  combination="(HLS OR Sobel X) AND ROI", records=records)
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Processed: {processed}/{len(paths)}; failed: {len(paths)-processed}")
    return report


if __name__ == "__main__":
    process_final_images()
