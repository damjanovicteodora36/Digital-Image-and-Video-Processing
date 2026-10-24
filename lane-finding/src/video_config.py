"""Tested resolution scaling for independent project-video processing."""
import math
import numpy as np

if __package__:
    from .thresholding import DEFAULT_ROI
    from .perspective_transform import SOURCE_POINTS,DESTINATION_POINTS
    from .lane_detection import DEFAULTS as SEARCH_DEFAULTS
    from .lane_geometry import DEFAULTS as GEOMETRY_DEFAULTS
else:
    from thresholding import DEFAULT_ROI
    from perspective_transform import SOURCE_POINTS,DESTINATION_POINTS
    from lane_detection import DEFAULTS as SEARCH_DEFAULTS
    from lane_geometry import DEFAULTS as GEOMETRY_DEFAULTS

REFERENCE_SIZE=(1280,720)
SUPPORTED_SIZES=((1280,720),(960,540))
CAMERA_ASSUMPTION=('The three project_video clips are assumed to share camera, lens and full field of view; '
                   'only resolution differs. This is an adopted project assumption, not independently verified data.')

def resolution_config(size):
    """Reproduce the tested settings; do not tune by video name or frame."""
    size=tuple(size)
    if size not in SUPPORTED_SIZES:raise ValueError(f'Unsupported resolution {size}; expected one of {SUPPORTED_SIZES}')
    width,height=size
    sx=(width-1)/(REFERENCE_SIZE[0]-1);sy=(height-1)/(REFERENCE_SIZE[1]-1)
    area=sx*sy
    return dict(image_size=size,reference_size=REFERENCE_SIZE,
        camera_scale=(width/REFERENCE_SIZE[0],height/REFERENCE_SIZE[1]),pixel_scale=(sx,sy),
        source_points=(np.asarray(SOURCE_POINTS)*[sx,sy]).tolist(),
        destination_points=(np.asarray(DESTINATION_POINTS)*[sx,sy]).tolist(),roi=DEFAULT_ROI,
        search=dict(nwindows=SEARCH_DEFAULTS['nwindows'],margin=round(SEARCH_DEFAULTS['margin']*sx),
            minpix=math.ceil(SEARCH_DEFAULTS['minpix']*area),min_lane_pixels=math.ceil(SEARCH_DEFAULTS['min_lane_pixels']*area),
            min_supported_windows=SEARCH_DEFAULTS['min_supported_windows'],
            seed_parameters=dict(min_peak=math.ceil(20*sy),peak_distance=round(60*sx))),
        fit=dict(min_pixels=math.ceil(300*area),min_unique_y=math.ceil(100*sy),min_supported_windows=4,
            max_rmse=40*sx,max_slope=2*sx/sy,max_horizontal_span=.35),
        geometry=GEOMETRY_DEFAULTS | dict(xm_per_pix=GEOMETRY_DEFAULTS['xm_per_pix']/sx,
            ym_per_pix=GEOMETRY_DEFAULTS['ym_per_pix']/sy,
            width_bounds_px=tuple(v*sx for v in GEOMETRY_DEFAULTS['width_bounds_px'])),
        large_jump_threshold_px=30*sx,camera_assumption=CAMERA_ASSUMPTION,
        camera_identity_independently_confirmed=False,temporal_tracking=False,
        coordinate_policy='Dimension-ratio camera intrinsics; endpoint-normalized selected road points/ROI. These are the tested policies, not an exact resampling equivalence claim.')

def scale_calibration(calibration,size):
    """Scale K only; reuse the same distortion under the project assumption."""
    resolution_config(size)
    matrix,distortion,reference=calibration
    if tuple(reference)!=REFERENCE_SIZE:raise ValueError(f'Expected reference calibration at {REFERENCE_SIZE}')
    sx,sy=size[0]/reference[0],size[1]/reference[1]
    return np.diag([sx,sy,1.])@matrix,distortion.copy(),tuple(size)
