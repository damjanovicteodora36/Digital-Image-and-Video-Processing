"""Sliding-window pixel candidates and reliability-gated quadratic fits."""

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .perspective_transform import perspective_matrices, warp_binary
    from .thresholding import load_calibration, undistort_image, binarize_image, save_image
else:
    from perspective_transform import perspective_matrices, warp_binary
    from thresholding import load_calibration, undistort_image, binarize_image, save_image

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULTS = dict(nwindows=9, margin=80, minpix=50, min_lane_pixels=300,
                min_supported_windows=4)


def validate_binary(binary):
    """Require a nonempty two-dimensional uint8 0/1 image."""
    if (not isinstance(binary, np.ndarray) or binary.dtype != np.uint8
            or binary.ndim != 2 or min(binary.shape, default=0) < 2
            or not np.isin(binary, (0, 1)).all()):
        raise ValueError("Expected a uint8 0/1 image with height and width >= 2")


def lower_half_histogram(binary):
    """Count active pixels by column in the lower half."""
    validate_binary(binary)
    return binary[binary.shape[0] // 2:].sum(axis=0, dtype=np.int64)


def find_start_positions(binary, left_range=(0.12, 0.45), right_range=(0.55, 0.88),
                         separation=(0.30, 0.70), min_peak=20,
                         ambiguity_ratio=0.85, peak_distance=60):
    """Search separated spatial bands; reject weak or competing peaks."""
    histogram = lower_half_histogram(binary)
    width = binary.shape[1]
    reasons, seeds = [], []
    for side, bounds in zip(("left", "right"), (left_range, right_range)):
        if not 0 <= bounds[0] < bounds[1] <= 1:
            raise ValueError("Search ranges must be normalized increasing intervals")
        low, high = int(bounds[0]*width), int(bounds[1]*width)
        values = histogram[low:high]
        if not len(values) or values.max() < min_peak:
            reasons.append(f"{side}: no peak with at least {min_peak} pixels")
            seeds.append(None)
            continue
        seed = low + int(np.argmax(values))
        competing = values.copy()
        competing[max(0, seed-low-peak_distance):min(len(values), seed-low+peak_distance+1)] = 0
        if competing.max(initial=0) >= ambiguity_ratio * histogram[seed]:
            reasons.append(f"{side}: competing distant histogram peak")
            seeds.append(None)
        else:
            seeds.append(seed)
    if all(seed is not None for seed in seeds):
        distance = (seeds[1]-seeds[0])/width
        if not separation[0] <= distance <= separation[1]:
            reasons.append("Seed separation outside expected lane-width interval")
            seeds = [None, None]
    return histogram, seeds, reasons


def sliding_window_search(binary, nwindows=9, margin=80, minpix=50,
                          min_lane_pixels=300, min_supported_windows=4, seed_parameters=None,
                          enable_fallback=True):
    """Return separate pixel coordinates, windows, and heuristic diagnostics."""
    validate_binary(binary)
    if (not all(isinstance(v, int) and v > 0 for v in
                (nwindows, margin, minpix, min_lane_pixels, min_supported_windows))
            or nwindows > binary.shape[0] or min_supported_windows > nwindows):
        raise ValueError("Invalid window parameters")
    histogram, seeds, reasons = find_start_positions(binary, **(seed_parameters or {}))
    y, x = binary.nonzero()
    centers = seeds.copy()
    indices = [[], []]
    windows, supported = [], [0, 0]
    height, width = binary.shape
    for level in range(nwindows):
        bottom = height - level * height // nwindows
        top = height - (level+1) * height // nwindows
        selections, boxes = [], []
        for center in centers:
            if center is None:
                selections.append(np.zeros(len(x), dtype=bool))
                boxes.append(None)
            else:
                lo, hi = max(0, center-margin), min(width, center+margin)
                selections.append((y >= top) & (y < bottom) & (x >= lo) & (x < hi))
                boxes.append([lo, top, hi, bottom])
        # Assign overlap to the closer center, preventing duplicate ownership.
        overlap = selections[0] & selections[1]
        if overlap.any():
            nearer_left = abs(x-centers[0]) <= abs(x-centers[1])
            selections[0][overlap & ~nearer_left] = False
            selections[1][overlap & nearer_left] = False
        counts = []
        for side, selected in enumerate(selections):
            found = np.flatnonzero(selected)
            indices[side].extend(found.tolist())
            counts.append(len(found))
            if len(found) >= minpix:
                supported[side] += 1
                centers[side] = int(np.mean(x[found]))
        windows.append(dict(boxes=boxes, counts=counts))
    lanes = {}
    for side, label in enumerate(("left", "right")):
        chosen = np.asarray(indices[side], dtype=np.int64)
        lanes[label] = dict(x=x[chosen], y=y[chosen], count=len(chosen), supported_windows=supported[side])
        if len(chosen) < min_lane_pixels:
            reasons.append(f"{label}: fewer than {min_lane_pixels} selected pixels")
        if supported[side] < min_supported_windows:
            reasons.append(f"{label}: fewer than {min_supported_windows} windows with >= {minpix} pixels")
    result = dict(histogram=histogram, seeds=seeds, lanes=lanes, windows=windows,
                passes_heuristics=not reasons, reasons=reasons,
                note="Passing heuristics does not establish correct lane identity; inspect visualization")
    result["initialization"] = [dict(method="primary" if s is not None else "none", x=s,
        band=[height//2, height], start_window=0 if s is not None else None,
        reason="Accepted lower-half histogram peak" if s is not None else "; ".join(reasons)) for s in seeds]
    if enable_fallback and any(s is None for s in seeds):
        _initialize_fallback(binary, result, nwindows, margin, minpix,
                             min_lane_pixels, min_supported_windows, seed_parameters or {})
    return result


def _initialize_fallback(binary, result, nwindows, margin, minpix, min_lane_pixels,
                         min_supported_windows, seed_parameters):
    """Seed missing sides in supported bands and follow independently both ways."""
    height, width = binary.shape
    y, x = binary.nonzero()
    ranges = [seed_parameters.get("left_range", (0.12, 0.45)),
              seed_parameters.get("right_range", (0.55, 0.88))]
    min_peak = seed_parameters.get("min_peak", 20)
    ratio = seed_parameters.get("ambiguity_ratio", 0.85)
    distance = seed_parameters.get("peak_distance", 60)
    separation = seed_parameters.get("separation", (0.30, 0.70))
    for side, label in enumerate(("left", "right")):
        if result["seeds"][side] is not None:
            continue
        candidates = []
        for level in range(nwindows):
            top, bottom = height-(level+1)*height//nwindows, height-level*height//nwindows
            hist = binary[top:bottom].sum(axis=0)
            low, high = int(ranges[side][0]*width), int(ranges[side][1]*width)
            values = hist[low:high].copy()
            if not len(values) or values.max() < min_peak:
                continue
            peak = low+int(np.argmax(values))
            strength = int(hist[peak])
            values[max(0, peak-low-distance):min(len(values), peak-low+distance+1)] = 0
            if values.max(initial=0) >= ratio*strength:
                continue
            local = (y >= top)&(y < bottom)&(abs(x-peak) < margin)
            if local.sum() < minpix:
                continue
            center = int(np.mean(x[local]))
            other = result["lanes"][("right", "left")[side]]
            in_band = (other["y"] >= top)&(other["y"] < bottom)
            reference = int(np.median(other["x"][in_band])) if in_band.any() else result["seeds"][1-side]
            # No unanchored single-side fallback when both primary seeds failed.
            if reference is None:
                continue
            gap = (reference-center if side == 0 else center-reference)/width
            if not separation[0] <= gap <= separation[1]:
                continue
            candidates.append((strength, center, level, top, bottom))
        if not candidates:
            result["initialization"][side]["reason"] = "No unambiguous locally supported, pair-compatible fallback band"
            continue
        candidates.sort(reverse=True)
        best = candidates[0]
        if any(abs(c[1]-best[1]) > distance and c[0] >= ratio*best[0] for c in candidates[1:]):
            result["initialization"][side]["reason"] = "Competing fallback positions across bands"
            continue
        _, seed, start, top, bottom = best
        result["seeds"][side] = seed
        result["initialization"][side] = dict(method="fallback", x=seed, band=[top,bottom],
            start_window=start, reason="Local peak and pixel support; unambiguous across bands; compatible with other boundary")
        chosen = []
        def visit(level, center):
            top, bottom = height-(level+1)*height//nwindows, height-level*height//nwindows
            low, high = max(0, center-margin), min(width, center+margin)
            selected = (y >= top)&(y < bottom)&(x >= low)&(x < high)
            other = result["lanes"][("right", "left")[side]]
            # Primary ownership is preserved; each pixel belongs to at most one side.
            occupied = np.zeros(binary.shape, bool)
            occupied[other["y"],other["x"]] = True
            selected &= ~occupied[y,x]
            found = np.flatnonzero(selected)
            chosen.extend(found.tolist())
            result["windows"][level]["boxes"][side] = [low,top,high,bottom]
            result["windows"][level]["counts"][side] = len(found)
            return int(np.mean(x[found])) if len(found) >= minpix else center
        origin = visit(start, seed)
        for levels in (range(start+1,nwindows), range(start-1,-1,-1)):
            center = origin
            for level in levels:
                center = visit(level, center)
        selected = np.array(chosen, dtype=np.int64)
        supported = sum(w["counts"][side] >= minpix for w in result["windows"])
        result["lanes"][label] = dict(x=x[selected],y=y[selected],count=len(selected),supported_windows=supported)
        result["reasons"] = [r for r in result["reasons"] if not r.startswith(label+":")]
        if len(selected) < min_lane_pixels:
            result["reasons"].append(f"{label}: fewer than {min_lane_pixels} selected pixels")
        if supported < min_supported_windows:
            result["reasons"].append(f"{label}: fewer than {min_supported_windows} windows with >= {minpix} pixels")
    result["passes_heuristics"] = not result["reasons"]


def save_visualizations(binary, result, output_dir, name):
    """Save windows, assigned pixels, and a histogram with search bands."""
    canvas = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)
    for lane, color in zip(result["lanes"].values(), ((255, 0, 0), (0, 0, 255))):
        canvas[lane["y"], lane["x"]] = color
    for window in result["windows"]:
        for box in window["boxes"]:
            if box is not None:
                lo, top, hi, bottom = box
                cv2.rectangle(canvas, (lo, top), (hi-1, bottom-1), (0, 255, 0), 2)
    cv2.putText(canvas, "Left: blue | Right: red | Windows: green", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    save_image(output_dir / f"{name}_windows.jpg", canvas)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result["histogram"], color="black")
    for bounds, color, seed, side in zip(((0.12, 0.45), (0.55, 0.88)),
                                        ("blue", "red"), result["seeds"], ("left", "right")):
        ax.axvspan(bounds[0]*binary.shape[1], bounds[1]*binary.shape[1], color=color, alpha=0.08)
        if seed is not None:
            ax.axvline(seed, color=color, label=f"{side} seed: {seed}")
    ax.set(xlabel="Column x [pixels]", ylabel="Active pixels in lower half",
           title=f"{name}: histogram and accepted seeds")
    if any(s is not None for s in result["seeds"]):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_histogram.png")
    plt.close(fig)


def fit_lane_polynomials(result, image_shape, min_pixels=300, min_unique_y=100,
                         min_supported_windows=4, max_rmse=40.0, max_slope=2.0,
                         max_horizontal_span=0.35):
    """Fit x=a*y**2+b*y+c independently, rejecting unreliable sides.

    Supported rows contain assigned pixels. Unsupported internal gaps are
    interpolation; rows outside the observed y range are extrapolation.
    Temporal callers supply measured supported_bands instead of windows;
    the same minimum support-count threshold applies to either search.
    Geometric limits are pixel-space heuristics, not proof of lane identity.
    """
    height, width = image_shape[:2]
    fits = {}
    for index, side in enumerate(("left", "right")):
        lane = result["lanes"][side]
        x, y = np.asarray(lane["x"]), np.asarray(lane["y"])
        reasons = [r for r in result["reasons"] if r.startswith(side+":") or not r.startswith(("left:", "right:"))]
        if result["seeds"][index] is None:
            reasons.append("No reliable starting position")
        if len(x) < min_pixels:
            reasons.append(f"Fewer than {min_pixels} pixels")
        if len(np.unique(y)) < min_unique_y:
            reasons.append(f"Fewer than {min_unique_y} distinct rows")
        # Temporal corridors have measured horizontal bands, not sliding windows.
        support_kind = "bands" if "supported_bands" in lane else "windows"
        support_count = lane.get("supported_bands", lane.get("supported_windows", 0))
        if support_count < min_supported_windows:
            reasons.append(f"Fewer than {min_supported_windows} supported {support_kind}")
        fit = dict(reliable=False, pixel_count=len(x), unique_rows=len(np.unique(y)),
                   supported_windows=lane.get("supported_windows"), coefficients=None,
                   reasons=reasons)
        if support_kind == "bands":
            fit["supported_bands"] = support_count
        if reasons:
            fits[side] = fit
            continue
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Pixel coordinates must be matching finite arrays")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", np.exceptions.RankWarning)
                coefficients = np.polyfit(y, x, 2)
            rows = np.arange(height)
            curve = np.polyval(coefficients, rows)
            rmse = float(np.sqrt(np.mean((np.polyval(coefficients, y)-x)**2)))
            if not np.isfinite(coefficients).all() or not np.isfinite(curve).all():
                reasons.append("Nonfinite fit")
            if np.any((curve < 0) | (curve >= width)):
                reasons.append("Curve leaves the image")
            if np.ptp(curve) > max_horizontal_span*width:
                reasons.append("Excessive horizontal span")
            if np.max(np.abs(2*coefficients[0]*rows+coefficients[1])) > max_slope:
                reasons.append("Excessive slope")
            if rmse > max_rmse:
                reasons.append("Excessive horizontal residual")
            fit["rmse_pixels"] = rmse
            if not reasons:
                supported = np.isin(rows, y)
                extrapolated = (rows < y.min()) | (rows > y.max())
                fit.update(reliable=True, coefficients=coefficients.tolist(), y=rows, x=curve,
                           supported_rows=supported, extrapolated_rows=extrapolated,
                           interpolated_rows=~supported & ~extrapolated)
        except (np.exceptions.RankWarning, np.linalg.LinAlgError, ValueError) as error:
            reasons.append(f"Numerical fit failure: {error}")
        fits[side] = fit
    return fits


def save_polynomial_visualization(binary, result, fits, output_dir, name):
    """Use solid curves on observed rows and dashed curves across missing rows."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(binary, cmap="gray", vmin=0, vmax=1)
    for side, color in (("left", "deepskyblue"), ("right", "orangered")):
        lane, fit = result["lanes"][side], fits[side]
        ax.scatter(lane["x"], lane["y"], s=0.3, color=color, alpha=0.4)
        if fit["reliable"]:
            for mask, style, label in [(fit["supported_rows"], "-", "observed rows"),
                                        (fit["interpolated_rows"], "--", "interpolation"),
                                        (fit["extrapolated_rows"], ":", "extrapolation")]:
                ax.plot(np.where(mask, fit["x"], np.nan), fit["y"], style,
                        color=color, linewidth=2, label=f"{side}: {label}")
        else:
            ax.plot([], [], color=color, label=f"{side}: UNRELIABLE - no fit")
    ax.set(xlim=(0, binary.shape[1]-1), ylim=(binary.shape[0]-1, 0),
           title=f"{name}: quadratic candidates (pixel coordinates)", xlabel="x [pixels]", ylabel="y [pixels]")
    ax.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_polynomial.png")
    plt.close(fig)


def main():
    """Run pixel search and reliability-gated fits on the three images."""
    output_dir = PROJECT_DIR / "results/lane_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration()
    matrix, _ = perspective_matrices()
    records = []
    for name in ("straight_lines1", "test1", "test3"):
        image = cv2.imread(str(PROJECT_DIR / "test_images" / f"{name}.jpg"))
        if image is None or image.shape[1::-1] != calibration[2] or image.shape[1::-1] != (1280, 720):
            raise ValueError(f"{name}: incompatible input for calibrated perspective geometry")
        binary = warp_binary(binarize_image(undistort_image(image, calibration)), matrix)
        result = sliding_window_search(binary, **DEFAULTS)
        fits = fit_lane_polynomials(result, binary.shape)
        save_polynomial_visualization(binary, result, fits, output_dir, name)
        fit_report = {side: {k: v for k, v in fit.items() if k not in
                            ("x", "y", "supported_rows", "extrapolated_rows", "interpolated_rows")}
                      for side, fit in fits.items()}
        (output_dir / f"{name}_polynomial.json").write_text(json.dumps(fit_report, indent=2)+"\n", encoding="utf-8")
        np.savez_compressed(output_dir / f"{name}_polynomial.npz",
                            **{f"{side}_{key}": fit[key] for side, fit in fits.items() if fit["reliable"]
                               for key in ("coefficients", "x", "y", "supported_rows", "extrapolated_rows", "interpolated_rows")})
        print("Polynomial fits:", json.dumps(fit_report))
        save_visualizations(binary, result, output_dir, name)
        np.savez_compressed(output_dir / f"{name}_pixels.npz", binary=binary,
                            **{f"{side}_{axis}": lane[axis] for side, lane in result["lanes"].items()
                               for axis in ("x", "y")})
        record = dict(image=name, seeds=result["seeds"], distortion_correction_applied=True,
                      lanes={side: dict(count=lane["count"], supported_windows=lane["supported_windows"])
                             for side, lane in result["lanes"].items()},
                      passes_heuristics=result["passes_heuristics"], reasons=result["reasons"],
                      windows=result["windows"])
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "windows"}))
    report = dict(parameters=DEFAULTS, seed_parameters=dict(left_range=(0.12, 0.45), right_range=(0.55, 0.88),
                  separation=(0.30, 0.70), min_peak=20, ambiguity_ratio=0.85, peak_distance=60),
                  caveat="Heuristic support only; visual verification of lane identity is required", records=records)
    (output_dir / "lane_detection_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
