"""Approximate lane geometry from accepted fits, using assumed metric scales."""

import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULTS = dict(ym_per_pix=30/720, xm_per_pix=3.7/640,
                width_bounds_px=(400, 900), max_width_variation_fraction=0.25,
                straight_epsilon=1e-9)


def curvature_radius(fit, height, ym_per_pix=30/720, xm_per_pix=3.7/640,
                     straight_epsilon=1e-9):
    """Return radius at the bottom row; infinity means practically straight."""
    if height < 2 or not all(np.isfinite(v) and v > 0 for v in
                              (ym_per_pix, xm_per_pix, straight_epsilon)):
        raise ValueError("Positive scales, tolerance, and height >= 2 required")
    if not fit["reliable"]:
        return dict(radius_m=None, status="unreliable_fit", reasons=fit.get("reasons", []))
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise ValueError("Expected three finite polynomial coefficients")
    a, b, c = coefficients * [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]
    y_m = (height-1)*ym_per_pix
    if abs(2*a) <= straight_epsilon:
        return dict(radius_m=float("inf"), status="practically_straight", metric_coefficients=[a, b, c])
    radius = float((1+(2*a*y_m+b)**2)**1.5 / abs(2*a))
    return dict(radius_m=radius, status="estimated", metric_coefficients=[a, b, c])


def compute_lane_geometry(fits, image_shape, observed_y_range=None, **parameters):
    """Gate offset by lane ordering and width; never invent an absent boundary."""
    settings = DEFAULTS | parameters
    height, width = image_shape[:2]
    low, high = settings["width_bounds_px"]
    if width < 2 or not 0 < low < high or not 0 <= settings["max_width_variation_fraction"] < 1:
        raise ValueError("Invalid image dimensions or geometry thresholds")
    radii = {side: curvature_radius(fits[side], height, settings["ym_per_pix"],
                                   settings["xm_per_pix"], settings["straight_epsilon"])
             for side in ("left", "right")}
    result = dict(curvature=radii, offset_m=None, lane_width_px=None, reasons=[],
                  geometry_accepted=False, vehicle_center_x_px=width/2,
                  evaluation_y_px=height-1)
    if any(not fits[side]["reliable"] for side in ("left", "right")):
        result["reasons"] = [f"{side}: unreliable boundary; offset not computed"
                             for side in ("left", "right") if not fits[side]["reliable"]]
        return result
    left, right = (np.asarray(fits[s]["coefficients"]) for s in ("left", "right"))
    difference = right-left

    def extrema(y0, y1):
        ys = [y0, y1]
        if abs(difference[0]) > 1e-15:
            vertex = -difference[1]/(2*difference[0])
            if y0 < vertex < y1:
                ys.append(vertex)
        return np.polyval(difference, ys)

    y0, y1 = (0, height-1) if observed_y_range is None else observed_y_range
    if not 0 <= y0 <= y1 <= height-1:
        raise ValueError("Invalid observed vertical range")
    widths = extrema(y0, y1)
    full_widths = extrema(0, height-1)
    bottom_left, bottom_right = float(np.polyval(left, height-1)), float(np.polyval(right, height-1))
    bottom_width = bottom_right-bottom_left
    variation = float((widths.max()-widths.min())/bottom_width) if bottom_width > 0 else None
    result.update(lane_width_px=bottom_width, left_bottom_x_px=bottom_left,
                  right_bottom_x_px=bottom_right, observed_y_range=[float(y0), float(y1)],
                  observed_width_min_px=float(widths.min()), observed_width_max_px=float(widths.max()),
                  width_variation_fraction=variation)
    if full_widths.min() <= 0:
        result["reasons"].append("Boundaries cross or reverse order in the image")
    if not low <= bottom_width <= high:
        result["reasons"].append("Bottom width outside assumed pixel bounds")
    if variation is None or variation > settings["max_width_variation_fraction"]:
        result["reasons"].append("Excessive width variation over observed vertical range")
    if not result["reasons"]:
        center = (bottom_left+bottom_right)/2
        result.update(geometry_accepted=True, lane_center_x_px=center,
                      offset_m=float((width/2-center)*settings["xm_per_pix"]))
    return result


def json_safe(value):
    """Represent infinity as null plus its explicit practically_straight status."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    source = PROJECT_DIR / "results/lane_detection"
    output = PROJECT_DIR / "results/lane_geometry"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("straight_lines1", "test1", "test3"):
        fits = json.loads((source / f"{name}_polynomial.json").read_text())
        with np.load(source / f"{name}_pixels.npz") as pixels:
            shape = pixels["binary"].shape
            ys = [pixels[f"{side}_y"] for side in ("left", "right")]
            observed = (max(int(y.min()) for y in ys), min(int(y.max()) for y in ys)) if all(len(y) for y in ys) else None
        geometry = compute_lane_geometry(fits, shape, observed)
        geometry["image"] = name
        geometry["fit_reliable"] = {s: fits[s]["reliable"] for s in fits}
        with np.load(source / f"{name}_polynomial.npz") as curves:
            geometry["bottom_support"] = {side: ("observed" if curves[f"{side}_supported_rows"][-1]
                else "extrapolated" if curves[f"{side}_extrapolated_rows"][-1] else "interpolated")
                for side in fits if fits[side]["reliable"]}
        records.append(geometry)
        canvas = np.full((470, 1280, 3), 245, np.uint8)
        lines = [f"{name}: APPROXIMATE geometry - assumed metric scales",
                 "xm_per_pix = 3.7/640; ym_per_pix = 30/720; camera x = width/2"]
        for side in ("left", "right"):
            item = geometry["curvature"][side]
            lines.append(f"{side}: R = {item['radius_m']} m; {item['status']}; bottom: {geometry['bottom_support'].get(side, 'unavailable')}")
        lines += [f"Bottom lane width: {geometry['lane_width_px']} px",
                  f"Vehicle offset: {geometry['offset_m']} m (+right / -left)",
                  "Checks: " + ("passed (heuristics only)" if geometry["geometry_accepted"] else "REJECTED")]
        lines.extend(geometry["reasons"])
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (20, 35+40*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (20, 20, 20), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output / f"{name}_geometry.png"), canvas):
            raise OSError("Could not save geometry summary")
        print(json.dumps(json_safe(geometry), allow_nan=False))
    report = dict(assumptions=DEFAULTS,
                  caveat="Approximate scales only, not metric calibration; camera assumed at vehicle center. Positive offset means vehicle right of lane center. Curvature may use an extrapolated bottom value. Infinity is serialized as null with practically_straight status.",
                  records=records)
    (output / "geometry_report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def uncertainty_labels(geometry, radius_ratio_threshold=3.0):
    """Return informational warnings without changing geometry acceptance."""
    if not np.isfinite(radius_ratio_threshold) or radius_ratio_threshold <= 1:
        raise ValueError("Radius ratio threshold must exceed one")
    labels = []
    for side, reliable in geometry["fit_reliable"].items():
        if not reliable:
            labels.append(f"{side}: UNRELIABLE - no curve or radius")
        elif geometry["bottom_support"].get(side) == "extrapolated":
            labels.append(f"{side}: bottom EXTRAPOLATED; radius and offset use this estimate")
    items = [geometry["curvature"][side] for side in ("left", "right")]
    if all(item["status"] in ("estimated", "practically_straight") for item in items):
        values = [float("inf") if item["status"] == "practically_straight" else item["radius_m"] for item in items]
        if ((np.isinf(values[0]) != np.isinf(values[1])) or
                (all(np.isfinite(v) and v > 0 for v in values)
                 and max(values)/min(values) >= radius_ratio_threshold)):
            labels.append(f"WARNING: boundary radii differ by >= {radius_ratio_threshold:g}x (informational only)")
    return labels


def generate_geometry_visualizations(radius_ratio_threshold=3.0):
    """Read saved JSON values; preserve existing reports and text summaries."""
    output = PROJECT_DIR / "results/lane_geometry"
    source = PROJECT_DIR / "results/lane_detection"
    report = json.loads((output / "geometry_report.json").read_text())
    for geometry in report["records"]:
        name = geometry["image"]
        with np.load(source / f"{name}_pixels.npz") as data:
            binary = data["binary"]
        height, width = binary.shape
        road = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR)
        colors = {"left": (255, 160, 0), "right": (0, 150, 255)}
        with np.load(source / f"{name}_polynomial.npz") as curves:
            for side, reliable in geometry["fit_reliable"].items():
                if not reliable:
                    continue
                x, y = curves[f"{side}_x"], curves[f"{side}_y"]
                observed = curves[f"{side}_supported_rows"]
                extrapolated = curves[f"{side}_extrapolated_rows"]
                for i in range(1, len(y)):
                    # Solid observed segments; dashed gaps; dotted extrapolation.
                    supported = observed[i-1] and observed[i]
                    draw = supported or (i % (12 if extrapolated[i] else 20) < (3 if extrapolated[i] else 10))
                    if draw:
                        cv2.line(road, (int(round(x[i-1])), int(y[i-1])),
                                 (int(round(x[i])), int(y[i])), colors[side], 3)
        camera = int(round(geometry["vehicle_center_x_px"]))
        cv2.drawMarker(road, (camera, height-12), (255, 0, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
        if geometry["geometry_accepted"] and geometry["offset_m"] is not None:
            center = int(round(geometry["lane_center_x_px"]))
            cv2.drawMarker(road, (center, height-35), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.line(road, (center, height-35), (center, height-1), (0, 255, 0), 1)
        lines = [f"{name}: APPROXIMATE - assumed scales, not metric calibration",
                 "Left: blue | Right: orange | solid: observed; dashed: gaps; dotted: extrapolation"]
        for side in ("left", "right"):
            item = geometry["curvature"][side]
            radius = ("infinite (practically straight)" if item["status"] == "practically_straight"
                      else "unavailable" if item["radius_m"] is None else f"~{item['radius_m']:.0f} m")
            lines.append(f"{side} radius: {radius}")
        offset = geometry["offset_m"]
        lines.append("Offset: unavailable; no accepted lane center" if offset is None else
                     f"Offset: {offset:+.2f} m (+right / -left); bottom width: {geometry['lane_width_px']:.1f} px")
        lines.extend(uncertainty_labels(geometry, radius_ratio_threshold))
        lines.extend(geometry["reasons"])
        lines.append(f"Assumptions: xm/pix={report['assumptions']['xm_per_pix']:.6f}; ym/pix={report['assumptions']['ym_per_pix']:.6f}")
        header = np.full((32*len(lines)+20, width, 3), 245, np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(header, text, (12, 27+32*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.57, (20, 20, 20), 1, cv2.LINE_AA)
        footer = np.full((75, width, 3), 245, np.uint8)
        footer_lines = [f"Magenta triangle: assumed camera x={geometry['vehicle_center_x_px']:.1f} px"]
        if geometry["geometry_accepted"]:
            footer_lines.append(f"Green cross: lane center x={geometry['lane_center_x_px']:.1f} px (bottom row)")
        else:
            footer_lines.append("Lane center and offset not displayed: geometry unavailable/rejected")
        for i, text in enumerate(footer_lines):
            cv2.putText(footer, text, (12, 27+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        target = output / f"{name}_geometry_visualization.png"
        if not cv2.imwrite(str(target), np.vstack([header, road, footer])):
            raise OSError(f"Could not save {target}")
        print(target)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize-only", action="store_true", help="Use saved report without rewriting existing results")
    args = parser.parse_args()
    if not args.visualize_only:
        main()
    generate_geometry_visualizations()
