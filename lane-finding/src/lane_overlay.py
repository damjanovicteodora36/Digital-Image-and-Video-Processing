"""Render accepted lane candidates on undistorted images for visual review."""
import json
from pathlib import Path
import cv2
import numpy as np

if __package__:
    from .thresholding import load_calibration, undistort_image, binarize_image, save_image
    from .perspective_transform import perspective_matrices, warp_binary
    from .lane_detection import sliding_window_search, fit_lane_polynomials
    from .lane_geometry import compute_lane_geometry, uncertainty_labels, json_safe
else:
    from thresholding import load_calibration, undistort_image, binarize_image, save_image
    from perspective_transform import perspective_matrices, warp_binary
    from lane_detection import sliding_window_search, fit_lane_polynomials
    from lane_geometry import compute_lane_geometry, uncertainty_labels, json_safe

ROOT = Path(__file__).resolve().parents[1]


def lane_polygon(fits, image_shape):
    """Return a closed-order polygon only for accepted, ordered curves."""
    height, width = image_shape[:2]
    if not all(fits[s]['reliable'] for s in ('left', 'right')):
        return None
    y = np.arange(height)
    left, right = [np.polyval(fits[s]['coefficients'], y) for s in ('left', 'right')]
    if (not np.isfinite([left, right]).all() or np.any(left >= right)
            or np.any(left < 0) or np.any(right > width-1)):
        raise ValueError('Invalid lane polygon')
    return np.rint(np.vstack([np.column_stack([left, y]),
                              np.column_stack([right[::-1], y[::-1]])])).astype(np.int32)


def inverse_warp(image, inverse, output_size):
    """Map a bird's-eye color layer to the undistorted camera frame."""
    return cv2.warpPerspective(image, inverse, tuple(output_size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def format_overlay_text(fits, geometry, nearly_straight_m=10000, centered_m=0.05):
    """Format display-only estimates without changing geometry or diagnostics."""
    if not np.isfinite(nearly_straight_m) or nearly_straight_m <= 0:
        raise ValueError('Nearly-straight display threshold must be positive and finite')
    if not np.isfinite(centered_m) or centered_m < 0:
        raise ValueError('Centered display threshold must be nonnegative and finite')
    lines = ['LANE DETECTION']
    accepted = all(fits[s]['reliable'] for s in ('left', 'right')) and geometry['geometry_accepted']
    if not accepted:
        return lines + ['Lane unavailable', 'Insufficient reliable lane boundaries']
    for side in ('left', 'right'):
        curvature = geometry['curvature'][side]
        radius = curvature['radius_m']
        if curvature['status'] == 'practically_straight' or (radius is not None and radius >= nearly_straight_m):
            label = 'Nearly straight'
        elif radius is None or not np.isfinite(radius):
            label = 'Unavailable'
        else:
            label = f'~{radius:,.0f} m'
        lines.append(f'{side.title()} curvature: {label}')
    offset = geometry['offset_m']
    position = ('Centered' if abs(offset) <= centered_m else
                f'{abs(offset):.2f} m {"right" if offset > 0 else "left"} of center')
    lines.append(f'Vehicle position: {position}')
    extrapolated = [s.title() for s in ('left', 'right')
                    if geometry.get('bottom_support', {}).get(s) == 'extrapolated']
    note = 'Approximate estimates'
    if extrapolated:
        note += ' | Both boundaries extrapolated' if len(extrapolated) == 2 else f' | {extrapolated[0]} boundary extrapolated'
    return lines + [note]


def draw_overlay_text(image, lines):
    """Fit a compact information panel to the image width and resolution."""
    output = image.copy()
    scale = min(image.shape[1] / 1280, image.shape[0] / 720)
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = max(6, round(16 * scale))
    font_scale = max(0.3, 0.75 * scale)
    longest = max(cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in lines)
    font_scale *= min(1, (image.shape[1] - 2 * padding) / max(longest, 1))
    text_height, baseline = cv2.getTextSize('Ag', font, font_scale, 1)[0][1], 5
    spacing = max(text_height + baseline + 4, round(32 * scale))
    panel_height = padding * 2 + spacing * len(lines)
    cv2.rectangle(output, (0, 0), (image.shape[1]-1, panel_height-1), (25, 25, 25), -1)
    for index, line in enumerate(lines):
        cv2.putText(output, line, (padding, padding + text_height + spacing * index),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def render_overlay(image, fits, geometry, inverse, alpha=0.3, *, nearly_straight_m=10000, centered_m=0.05):
    """Return annotated image and whether a lane area was rendered."""
    if not 0 <= alpha <= 1:
        raise ValueError('Alpha must be in [0, 1]')
    output = image.copy()
    accepted = all(fits[s]['reliable'] for s in ('left', 'right')) and geometry['geometry_accepted']
    lines = format_overlay_text(fits, geometry, nearly_straight_m, centered_m)
    if accepted:
        polygon = lane_polygon(fits, image.shape)
        layer = np.zeros_like(image)
        cv2.fillPoly(layer, [polygon], (0, 255, 0))
        color = inverse_warp(layer, inverse, image.shape[1::-1])
        mask = np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(mask,[polygon],255)
        coverage = inverse_warp(mask,inverse,image.shape[1::-1]).astype(float)/255
        output = np.clip(image.astype(float)*(1-alpha*coverage[:,:,None]) + alpha*color,0,255).astype(np.uint8)
        edges = np.zeros_like(image)
        height = image.shape[0]
        cv2.polylines(edges,[polygon[:height]],False,(255,160,0),5)
        cv2.polylines(edges,[polygon[height:]],False,(0,120,255),5)
        edges = inverse_warp(edges,inverse,image.shape[1::-1])
        selected = edges.max(axis=2)>0
        output[selected] = edges[selected]
    # Text is drawn after compositing and never changes the numeric masks.
    return draw_overlay_text(output, lines), accepted


def main():
    out=ROOT/'results/lane_overlay'
    out.mkdir(parents=True,exist_ok=True)
    calibration=load_calibration(); matrix,inverse=perspective_matrices()
    records=[]; thumbnails=[]
    for path in sorted((ROOT/'test_images').glob('*.jpg')):
        image=cv2.imread(str(path))
        if image is None or image.shape[1::-1]!=(1280,720) or image.shape[1::-1]!=calibration[2]:
            continue
        prepared=undistort_image(image,calibration)
        warped=warp_binary(binarize_image(prepared),matrix)
        search=sliding_window_search(warped)
        fits=fit_lane_polynomials(search,warped.shape)
        ys=[search['lanes'][s]['y'] for s in ('left','right')]
        observed=(max(int(y.min()) for y in ys),min(int(y.max()) for y in ys)) if all(len(y) for y in ys) else None
        geometry=compute_lane_geometry(fits,warped.shape,observed)
        geometry['fit_reliable']={s:fits[s]['reliable'] for s in fits}
        geometry['bottom_support']={s:('observed' if f['supported_rows'][-1] else
            'extrapolated' if f['extrapolated_rows'][-1] else 'interpolated') for s,f in fits.items() if f['reliable']}
        overlay,rendered=render_overlay(prepared,fits,geometry,inverse)
        target=out/(path.stem+'_overlay.jpg'); save_image(target,overlay)
        thumb=cv2.resize(overlay,(640,360)); cv2.putText(thumb,path.stem,(10,345),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
        thumbnails.append(thumb)
        records.append(dict(image=path.name,output=target.name,lane_rendered=rendered,
            distortion_correction_applied=True,geometry=geometry,initialization=search['initialization'],
            fits={s:{k:v for k,v in f.items() if k not in ('x','y','supported_rows','interpolated_rows','extrapolated_rows')} for s,f in fits.items()}))
        print(path.name, 'lane rendered' if rendered else 'warning only')
    save_image(out/'overview.jpg',np.vstack([np.hstack(thumbnails[i:i+2]) for i in range(0,len(thumbnails),2)]))
    report=dict(total=len(records),rendered=sum(r['lane_rendered'] for r in records),
                alpha=0.3,frame='undistorted camera image',review_status='Awaiting visual acceptance',records=records)
    (out/'overlay_report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
