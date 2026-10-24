"""Resolution-aware independent-frame pipeline; temporal statistics never guide detection."""
import json
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

if __package__:
    from .thresholding import load_calibration, undistort_image, binarize_image, save_image
    from .perspective_transform import perspective_matrices, warp_binary
    from .lane_detection import sliding_window_search, fit_lane_polynomials
    from .lane_geometry import compute_lane_geometry, uncertainty_labels, json_safe
    from .lane_overlay import render_overlay
    from .video_config import resolution_config,scale_calibration,CAMERA_ASSUMPTION
else:
    from thresholding import load_calibration, undistort_image, binarize_image, save_image
    from perspective_transform import perspective_matrices, warp_binary
    from lane_detection import sliding_window_search, fit_lane_polynomials
    from lane_geometry import compute_lane_geometry, uncertainty_labels, json_safe
    from lane_overlay import render_overlay
    from video_config import resolution_config,scale_calibration,CAMERA_ASSUMPTION

ROOT=Path(__file__).resolve().parents[1]


def make_context(size=(1280,720)):
    """Cache only fixed calibration and transforms, never previous detections."""
    settings=resolution_config(size)
    matrix,inverse=perspective_matrices(settings['source_points'],settings['destination_points'])
    return scale_calibration(load_calibration(),size),matrix,inverse


def process_frame(frame, context=None):
    """Return a BGR overlay and diagnostics, independently of all other frames."""
    if (not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3
            or frame.shape[2]!=3):
        raise ValueError('Expected a uint8 BGR frame')
    settings=resolution_config(frame.shape[1::-1])
    calibration,matrix,inverse=make_context(frame.shape[1::-1]) if context is None else context
    if frame.shape[1::-1]!=calibration[2]:
        raise ValueError('Frame resolution incompatible with calibration')
    prepared=undistort_image(frame,calibration)
    binary=warp_binary(binarize_image(prepared,vertices=settings['roi']),matrix)
    search=sliding_window_search(binary,**settings['search'])
    fits=fit_lane_polynomials(search,binary.shape,**settings['fit'])
    ys=[search['lanes'][s]['y'] for s in ('left','right')]
    observed=(max(int(y.min()) for y in ys),min(int(y.max()) for y in ys)) if all(len(y) for y in ys) else None
    if observed is not None and observed[0]>observed[1]:observed=None
    geometry=compute_lane_geometry(fits,binary.shape,observed,**settings['geometry'])
    geometry['fit_reliable']={s:f['reliable'] for s,f in fits.items()}
    geometry['bottom_support']={s:('observed' if f['supported_rows'][-1] else
        'extrapolated' if f['extrapolated_rows'][-1] else 'interpolated') for s,f in fits.items() if f['reliable']}
    rendered,accepted=render_overlay(prepared,fits,geometry,inverse)
    diagnostics=dict(image_size=frame.shape[1::-1],undistortion_applied=True,
        camera_identity_assumed=True,camera_identity_independently_confirmed=False,
        two_accepted_boundaries=all(f['reliable'] for f in fits.values()),
        lane_rendered=accepted,geometry=geometry,initialization=search['initialization'],
        warnings=uncertainty_labels(geometry),search_reasons=search['reasons'],
        fits={s:{k:v for k,v in f.items() if k not in ('x','y','supported_rows','extrapolated_rows','interpolated_rows')} for s,f in fits.items()})
    return rendered,diagnostics


def video_info(path):
    capture=cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened(): raise OSError(f'Cannot open video: {path}')
        return dict(path=str(path),width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally: capture.release()


def verify_video(path,expected):
    info=video_info(path)
    if (info['width'],info['height'])!=(expected['width'],expected['height']) or not np.isclose(info['fps'],expected['fps'],atol=0.001):
        raise OSError('Output video dimensions or FPS mismatch')
    capture=cv2.VideoCapture(str(path)); count=0
    try:
        while True:
            ok,frame=capture.read()
            if not ok: break
            if frame.shape[:2]!=(expected['height'],expected['width']): raise OSError('Invalid decoded frame size')
            count+=1
    finally: capture.release()
    if count!=expected['frames']: raise OSError(f'Decoded {count} frames; expected {expected["frames"]}')
    return dict(**info,decoded_frames=count)


def process_video(input_path,output_dir=None):
    """Write to a new output directory, preserving every previous result."""
    input_path=Path(input_path)
    output_dir=ROOT/'results/video/final'/input_path.stem if output_dir is None else Path(output_dir)
    info=video_info(input_path)
    size=(info['width'],info['height']);settings=resolution_config(size)
    context=make_context(size)
    if not np.isfinite(info['fps']) or info['fps']<=0: raise ValueError('Invalid source FPS')
    target=output_dir/(input_path.stem+'_final.mp4')
    if target.resolve()==input_path.resolve(): raise ValueError('Output must differ from source')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}. Choose a new directory to preserve earlier results.')
    output_dir.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(input_path))
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),info['fps'],(info['width'],info['height']))
    if not writer.isOpened():
        capture.release(); writer.release(); raise OSError('Cannot initialize MP4 writer')
    records=[]; samples=[]; failures=[]; start_failure=None; previous=None; deltas=[]; warning_counts=Counter()
    largest_delta=-1; wall_start=time.perf_counter()
    try:
        while True:
            ok,frame=capture.read()
            if not ok: break
            begin=time.perf_counter(); result,diagnostics=process_frame(frame,context)
            elapsed=time.perf_counter()-begin; index=len(records)
            diagnostics.update(frame=index,processing_seconds=elapsed)
            for warning in diagnostics['warnings']+diagnostics['geometry']['reasons']: warning_counts[warning]+=1
            success=diagnostics['lane_rendered']
            center=diagnostics['geometry'].get('lane_center_x_px') if success else None
            delta=None
            if previous is not None and center is not None:
                delta=float(center-previous); deltas.append(delta)
            previous=center
            diagnostics['adjacent_center_delta_px']=delta
            if not success and start_failure is None: start_failure=index
            if success and start_failure is not None:
                failures.append(dict(start=start_failure,end=index-1,frames=index-start_failure)); start_failure=None
            category='success' if success else 'failure'
            if not any(s['category']==category for s in samples):
                name=f'{category}_{index:06d}.jpg'; save_image(output_dir/name,result); samples.append(dict(category=category,frame=index,path=name))
            if delta is not None and abs(delta)>largest_delta:
                largest_delta=abs(delta)
                save_image(output_dir/'largest_jump_after.jpg',result)
                # Previous overlay is retained only for diagnostics, never for detection.
                save_image(output_dir/'largest_jump_before.jpg',previous_overlay)
                jump=dict(before_frame=index-1,after_frame=index,delta_px=delta)
            if index%max(1,info['frames']//4)==0:
                name=f'interval_{index:06d}.jpg'; save_image(output_dir/name,result); samples.append(dict(category='interval',frame=index,path=name))
            previous_overlay=result
            writer.write(result); records.append(diagnostics)
            if len(records)%200==0: print(f'Processed {len(records)}/{info["frames"]}',flush=True)
    finally:
        capture.release(); writer.release()
    wall_seconds=time.perf_counter()-wall_start
    if not records: raise OSError('No decodable input frames')
    if len(records)!=info['frames']: raise OSError('Source decoding stopped before advertised frame count')
    if start_failure is not None: failures.append(dict(start=start_failure,end=len(records)-1,frames=len(records)-start_failure))
    for run in failures: run['seconds']=run['frames']/info['fps']
    checked=verify_video(target,dict(info,frames=len(records)))
    times=[r['processing_seconds'] for r in records]
    absolute=np.abs(deltas)
    report=dict(input=info,output=checked,total_frames=len(records),
        frames_two_accepted_boundaries=sum(r['two_accepted_boundaries'] for r in records),
        frames_with_lane=sum(r['lane_rendered'] for r in records),
        frames_without_reliable_lane=sum(not r['lane_rendered'] for r in records),
        fallback_frames=sum(any(i['method']=='fallback' for i in r['initialization']) for r in records),
        fallback_boundary_activations=sum(i['method']=='fallback' for r in records for i in r['initialization']),
        fallback_attempt_frames=sum(any(i['method']!='primary' for i in r['initialization']) for r in records),
        mean_processing_seconds=float(np.mean(times)),effective_processing_fps=len(records)/sum(times),
        wall_seconds=wall_seconds,end_to_end_fps=len(records)/wall_seconds,
        adjacent_center_change=dict(pairs=len(deltas),definition='Bottom lane-center x, adjacent frames with accepted geometry only; no bridging failures',
            mean_absolute_px=float(np.mean(absolute)) if len(deltas) else None,
            p95_absolute_px=float(np.percentile(absolute,95)) if len(deltas) else None,
            max_absolute_px=float(np.max(absolute)) if len(deltas) else None,
            over_30_px=int(np.count_nonzero(absolute>30))),
        largest_jump=jump if largest_delta>=0 else None,failure_runs=failures,
        longest_failure_frames=max((f['frames'] for f in failures),default=0),
        warning_frame_counts=dict(warning_counts),samples=samples,
        assumptions='No temporal state in detection. Metric scales are assumptions. Acceptance is not accuracy. Output is silent; source audio is not copied.',
        frames=records)
    sx=settings['pixel_scale'][0]
    report['adjacent_center_change'].update(large_jump_threshold_px=settings['large_jump_threshold_px'],
        large_jumps=int(np.count_nonzero(absolute>settings['large_jump_threshold_px'])),
        p95_reference_px=float(np.percentile(absolute/sx,95)) if len(absolute) else None,
        max_reference_px=float(np.max(absolute/sx)) if len(absolute) else None)
    report.update(resolution_configuration=settings,applied_camera_matrix=context[0][0].tolist(),
        distortion_coefficients=context[0][1].tolist(),camera_assumption=CAMERA_ASSUMPTION,
        camera_identity_independently_confirmed=False,undistortion_applied=True,
        extrapolated_bottom_frames={s:sum(r['geometry'].get('bottom_support',{}).get(s)=='extrapolated' for r in records) for s in ('left','right')})
    (output_dir/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"{input_path.name}: {len(records)} frames, {report['frames_with_lane']} accepted, "
          f"{report['frames_without_reliable_lane']} rejected; output: {target}",flush=True)
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='?',type=Path)
    parser.add_argument('--all-project-videos',action='store_true',help='Process project_video01,02,03 with the shared-camera assumption')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/video/final',help='Root directory for separate per-video outputs')
    args=parser.parse_args()
    if args.input is not None and args.all_project_videos:parser.error('Choose a single input or --all-project-videos')
    sources=[ROOT/'test_videos'/f'project_video{i:02d}.mp4' for i in (1,2,3)] if args.all_project_videos else [args.input or ROOT/'test_videos/project_video03.mp4']
    # Preflight all files and destinations before starting a batch.
    for source in sources:
        info=video_info(source);resolution_config((info['width'],info['height']))
        folder=args.output_dir/source.stem
        if folder.exists() and any(folder.iterdir()):raise FileExistsError(f'Output directory is not empty: {folder}')
    summary_path=args.output_dir/'summary.json'
    if summary_path.exists():raise FileExistsError(f'Summary already exists: {summary_path}')
    reports=[process_video(source,args.output_dir/source.stem) for source in sources]
    summary=dict(camera_assumption=CAMERA_ASSUMPTION,camera_identity_independently_confirmed=False,
        videos=[{k:v for k,v in r.items() if k not in ('frames','samples','failure_runs')} for r in reports])
    summary_path.write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()
