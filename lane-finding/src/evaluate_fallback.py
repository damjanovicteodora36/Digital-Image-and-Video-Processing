"""Controlled before/after evaluation, without overwriting the baseline report."""
import json
from pathlib import Path
import cv2
import numpy as np
from src.thresholding import load_calibration, undistort_image, binarize_image, save_image
from src.perspective_transform import perspective_matrices, warp_binary, mark_points
from src.lane_detection import sliding_window_search, fit_lane_polynomials, save_visualizations, save_polynomial_visualization
from src.lane_geometry import compute_lane_geometry, json_safe

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'results/evaluation/fallback_experiment'
    out.mkdir(parents=True,exist_ok=True)
    calibration=load_calibration(); matrix,_=perspective_matrices()
    records=[]
    for path in sorted((ROOT/'test_images').glob('*.jpg')):
        image=cv2.imread(str(path))
        if image is None or image.shape[1::-1] != (1280,720): continue
        prepared=undistort_image(image,calibration)
        binary=warp_binary(binarize_image(prepared),matrix)
        versions={}; searches={}
        for version,enabled in [('before',False),('after',True)]:
            search=sliding_window_search(binary,enable_fallback=enabled)
            searches[version]=search
            fits=fit_lane_polynomials(search,binary.shape)
            ys=[search['lanes'][s]['y'] for s in ('left','right')]
            observed=(max(int(y.min()) for y in ys),min(int(y.max()) for y in ys)) if all(len(y) for y in ys) else None
            geometry=compute_lane_geometry(fits,binary.shape,observed)
            summary={s:{k:v for k,v in f.items() if k not in ('x','y','supported_rows','interpolated_rows','extrapolated_rows')} for s,f in fits.items()}
            versions[version]=dict(initialization=search['initialization'],fits=summary,geometry=geometry,search_reasons=search['reasons'])
            save_visualizations(binary,search,out,path.stem+'_'+version)
            save_polynomial_visualization(binary,search,fits,out,path.stem+'_'+version)
            np.savez_compressed(out/(path.stem+'_'+version+'_pixels.npz'),**{s+'_'+a:l[a] for s,l in search['lanes'].items() for a in ('x','y')})
        unchanged=all(np.array_equal(searches['before']['lanes'][s][a],searches['after']['lanes'][s][a]) for s in ('left','right') for a in ('x','y'))
        records.append(dict(image=path.name,versions=versions,assigned_pixels_unchanged=unchanged,
            visual_identity_review=("pending" if path.stem in ('test2','test6') else "unchanged assignments verified")))
        panels=[cv2.imread(str(out/(path.stem+'_'+v+'_windows.jpg'))) for v in ('before','after')]
        save_image(out/(path.stem+'_before_after.jpg'),np.hstack(panels))
        if path.stem in ('test2','test6'):
            # Map assigned pixels back for visual identity checks only; no lane overlay.
            after=searches['after']['lanes']['right']
            inverse=np.linalg.inv(matrix)
            pts=np.column_stack([after['x'],after['y']]).astype(np.float32)
            mapped=cv2.perspectiveTransform(pts[None],inverse)[0]
            overlay=prepared.copy()
            for x,y in np.rint(mapped).astype(int):
                if 0<=x<1280 and 0<=y<720: overlay[y,x]=(0,0,255)
            save_image(out/(path.stem+'_right_pixel_identity.jpg'),overlay)
    baseline=json.loads((ROOT/'results/evaluation/evaluation_report.json').read_text())
    for record in records:
        old=next(r for r in baseline['records'] if r['image']==record['image'])
        for s in ('left','right'):
            for key in ('pixel_count','supported_windows','reliable','coefficients'):
                assert old['fits'][s][key]==record['versions']['before']['fits'][s][key]
    report=dict(conditions='Unchanged thresholding, perspective, fitting and geometry gates; fallback only for missing primary seeds',
                fallback='Window-aligned bands; same peak>=20, ambiguity ratio 0.85, local pixels>=50, separation 0.30..0.70 width; distinct cross-band candidates rejected. Requires another anchored boundary.',
                records=records)
    (out/'comparison_report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['| Image | Seeds before -> after L/R | Windows before -> after L/R | Pixels before -> after L/R | Accepted before -> after L/R | RMSE before -> after L/R | Width before -> after px | Geometry before -> after |', '|---|---|---|---|---|---|---|---|']
    for r in records:
        b,a=(r['versions'][v] for v in ('before','after'))
        pair=lambda v,key:'/'.join(str(v['fits'][s].get(key,'-')) for s in ('left','right'))
        seeds=lambda v:'/'.join(str(i['x']) for i in v['initialization'])
        lines.append(f"| {r['image']} | {seeds(b)} -> {seeds(a)} | {pair(b,'supported_windows')} -> {pair(a,'supported_windows')} | {pair(b,'pixel_count')} -> {pair(a,'pixel_count')} | {pair(b,'reliable')} -> {pair(a,'reliable')} | {pair(b,'rmse_pixels')} -> {pair(a,'rmse_pixels')} | {b['geometry']['lane_width_px']} -> {a['geometry']['lane_width_px']} | {b['geometry']['geometry_accepted']} -> {a['geometry']['geometry_accepted']} |")
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__': main()
