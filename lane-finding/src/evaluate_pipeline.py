"""Evaluate unchanged pipeline defaults; acceptance rate is not accuracy."""

import inspect
import json
from pathlib import Path
import cv2
import numpy as np

from src.thresholding import load_calibration, undistort_image, binarize_image, save_image, DEFAULT_PARAMETERS, DEFAULT_ROI
from src.perspective_transform import perspective_matrices, warp_binary, SOURCE_POINTS, DESTINATION_POINTS
from src.lane_detection import sliding_window_search, fit_lane_polynomials, save_visualizations, save_polynomial_visualization, DEFAULTS
from src.lane_geometry import compute_lane_geometry, uncertainty_labels, json_safe, DEFAULTS as GEOMETRY_DEFAULTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration()
    matrix, _ = perspective_matrices()
    records, skipped, panels = [], [], []
    for path in sorted((ROOT / "test_images").glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None or image.shape[1::-1] != (1280, 720) or image.shape[1::-1] != calibration[2]:
            skipped.append(path.name)
            continue
        name = path.stem
        prepared = undistort_image(image, calibration)
        binary = binarize_image(prepared)
        warped = warp_binary(binary, matrix)
        result = sliding_window_search(warped, **DEFAULTS)
        fits = fit_lane_polynomials(result, warped.shape)
        ys = [result["lanes"][s]["y"] for s in ("left", "right")]
        observed = (max(int(y.min()) for y in ys), min(int(y.max()) for y in ys)) if all(len(y) for y in ys) else None
        geometry = compute_lane_geometry(fits, warped.shape, observed)
        geometry["fit_reliable"] = {s: fits[s]["reliable"] for s in fits}
        geometry["bottom_support"] = {s: "observed" if fit["supported_rows"][-1] else
                                      "extrapolated" if fit["extrapolated_rows"][-1] else "interpolated"
                                      for s, fit in fits.items() if fit["reliable"]}
        warnings = uncertainty_labels(geometry)
        flags = dict(boundary_failure=[s for s in fits if not fits[s]["reliable"]],
                     radius_disagreement=any("radii differ" in w for w in warnings),
                     excessive_width_variation=any("width variation" in w for w in geometry["reasons"]),
                     bottom_extrapolation=[s for s, v in geometry["bottom_support"].items() if v == "extrapolated"],
                     possible_misidentification=("heuristic concern; manual review required" if
                         geometry["reasons"] or warnings or result["reasons"] else "not established; manual review required"))
        fit_summary = {s: {k: v for k, v in fit.items() if k not in
                          ("x", "y", "supported_rows", "interpolated_rows", "extrapolated_rows")}
                       for s, fit in fits.items()}
        save_image(output / f"{name}_undistorted.jpg", prepared)
        save_image(output / f"{name}_binary.png", binary*255)
        save_image(output / f"{name}_warped.png", warped*255)
        save_visualizations(warped, result, output, name)
        save_polynomial_visualization(warped, result, fits, output, name)
        np.savez_compressed(output / f"{name}_masks.npz", binary=binary, warped=warped)
        geometry_lines = [name+": approximate geometry", f"Width px: {geometry['lane_width_px']}",
                          f"Offset m: {geometry['offset_m']}"]
        geometry_lines += [f"{s} radius m: {geometry['curvature'][s]['radius_m']}" for s in fits]
        geometry_lines += geometry["reasons"] + warnings
        canvas = np.full((40*(len(geometry_lines)+1), 1280, 3), 245, np.uint8)
        for i, line in enumerate(geometry_lines):
            cv2.putText(canvas, line, (10, 30+40*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 1, cv2.LINE_AA)
        save_image(output / f"{name}_geometry.png", canvas)
        record = dict(image=path.name, distortion_correction_applied=True, fits=fit_summary,
                      seeds=result["seeds"], search_reasons=result["reasons"], geometry=geometry,
                      flags=flags, warnings=warnings, manual_review="pending")
        records.append(record)
        windows = cv2.imread(str(output / f"{name}_windows.jpg"))
        panel = np.hstack([cv2.resize(prepared,(640,360)),cv2.resize(windows,(640,360))])
        cv2.putText(panel,name,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        panels.append(panel)
    count = sum(all(f["reliable"] for f in r["fits"].values()) for r in records)
    report = dict(total=len(records), two_accepted_boundaries=count,
                  acceptance_fraction=count/len(records), not_accuracy="No manually annotated ground truth; heuristic acceptance only",
                  settings=dict(thresholding=DEFAULT_PARAMETERS, roi=DEFAULT_ROI, source=SOURCE_POINTS,
                                destination=DESTINATION_POINTS, windows=DEFAULTS, geometry=GEOMETRY_DEFAULTS,
                                fitting={k:v.default for k,v in inspect.signature(fit_lane_polynomials).parameters.items() if v.default is not inspect.Parameter.empty}),
                  skipped_incompatible=skipped, records=records)
    (output / "evaluation_report.json").write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save_image(output / "overview.jpg",np.vstack(panels))
    lines = ["| Image | Pixels L/R | Windows L/R | Fits L/R | RMSE L/R px | Width px | Radii L/R m | Offset m | Flags |",
             "|---|---:|---:|---|---|---:|---|---:|---|"]
    def number(v):
        return "-" if v is None else f"{v:.2f}"
    for r in records:
        f,g=r["fits"],r["geometry"]
        pair=lambda key: '/'.join(str(f[s].get(key,'-')) for s in ('left','right'))
        lines.append(f"| {r['image']} | {pair('pixel_count')} | {pair('supported_windows')} | {pair('reliable')} | {'/'.join(number(f[s].get('rmse_pixels')) for s in ('left','right'))} | {number(g['lane_width_px'])} | {'/'.join(number(g['curvature'][s]['radius_m']) for s in ('left','right'))} | {number(g['offset_m'])} | {'; '.join(r['warnings']+g['reasons'])} |")
    lines += ["",f"Two accepted boundaries: {count}/{len(records)}. This is not an accuracy estimate."]
    (output / "summary.md").write_text('\n'.join(lines)+'\n',encoding="utf-8")
    print('\n'.join(lines))


if __name__ == "__main__":
    main()
