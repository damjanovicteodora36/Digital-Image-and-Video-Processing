"""Experimental current-pixel tracking; no smoothing or stale lane rendering."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from .lane_detection import validate_binary, sliding_window_search, fit_lane_polynomials
from .lane_geometry import compute_lane_geometry, uncertainty_labels
from .lane_overlay import render_overlay
from .thresholding import undistort_image, binarize_image
from .perspective_transform import warp_binary
from .video_pipeline import make_context

@dataclass(frozen=True)
class TrackingConfig:
    margin: int = 80
    bands: int = 9
    min_band_pixels: int = 50
    min_band_rows: int = 10
    min_supported_bands: int = 4
    max_position_change_px: float = 60.0
    max_shape_change_px: float = 40.0
    temporal_samples: int = 9

    def __post_init__(self):
        if (any(not isinstance(v,int) or v<=0 for v in (self.margin,self.bands,self.min_band_pixels,self.min_band_rows,self.min_supported_bands,self.temporal_samples))
                or self.min_supported_bands>self.bands or self.temporal_samples<3
                or not all(np.isfinite(v) and v>0 for v in (self.max_position_change_px,self.max_shape_change_px))):
            raise ValueError('Invalid tracking configuration')

def band_support(x,y,height,config):
    """Measure actual pixel and distinct-row support in nonoverlapping bands."""
    bins=[]
    for level in range(config.bands):
        top=height-(level+1)*height//config.bands;bottom=height-level*height//config.bands
        selected=(y>=top)&(y<bottom)
        count=int(selected.sum());rows=len(np.unique(y[selected]))
        bins.append(dict(top=top,bottom=bottom,pixels=count,rows=rows,
            supported=count>=config.min_band_pixels and rows>=config.min_band_rows))
    return bins

def search_around_polynomials(binary,previous,config=TrackingConfig()):
    """Assign only current active pixels; ties go to the closer predicted curve."""
    validate_binary(binary)
    if config.bands>binary.shape[0]:raise ValueError('Too many support bands')
    if set(previous)!={'left','right'}:raise ValueError('A previous pair is required')
    y,x=np.nonzero(binary)
    curves=[]
    for side in ('left','right'):
        coefficients=np.asarray(previous[side],float)
        if coefficients.shape!=(3,) or not np.isfinite(coefficients).all():raise ValueError('Invalid previous coefficients')
        curves.append(np.polyval(coefficients,y))
    distances=[abs(x-curve) for curve in curves]
    masks=[d<=config.margin for d in distances]
    overlap=masks[0]&masks[1]
    nearer_left=distances[0]<=distances[1]
    masks[0][overlap&~nearer_left]=False;masks[1][overlap&nearer_left]=False
    lanes={}
    for side,selected in zip(('left','right'),masks):
        xx,yy=x[selected],y[selected];bins=band_support(xx,yy,binary.shape[0],config)
        lanes[side]=dict(x=xx,y=yy,count=len(xx),supported_bands=sum(b['supported'] for b in bins),support_bands=bins)
    return dict(lanes=lanes,seeds=[float(np.polyval(previous[s],binary.shape[0]-1)) for s in ('left','right')],
        reasons=[],windows=[],initialization=[dict(method='previous_polynomial',margin=config.margin) for _ in range(2)],
        note='Seeds are predicted search anchors, not histogram maxima; bands count actual current evidence')

def evaluate_search(search,shape,config=TrackingConfig()):
    """Reuse original fitting and geometry, adding search-independent support."""
    fits=fit_lane_polynomials(search,shape)
    reasons=[];support={}
    for side in ('left','right'):
        lane=search['lanes'][side];bins=band_support(lane['x'],lane['y'],shape[0],config)
        support[side]=bins
        if sum(b['supported'] for b in bins)<config.min_supported_bands:
            reasons.append(side+': insufficient spatial support in measured horizontal bands')
    ys=[search['lanes'][s]['y'] for s in ('left','right')]
    observed=(max(int(y.min()) for y in ys),min(int(y.max()) for y in ys)) if all(len(y) for y in ys) else None
    # The original geometry routine only uses observed_y_range for accepted fits.
    if observed is not None and observed[0]>observed[1]:observed=None
    geometry=compute_lane_geometry(fits,shape,observed)
    geometry['fit_reliable']={s:f['reliable'] for s,f in fits.items()}
    geometry['bottom_support']={s:('observed' if f['supported_rows'][-1] else 'extrapolated' if f['extrapolated_rows'][-1] else 'interpolated') for s,f in fits.items() if f['reliable']}
    reasons+=geometry['reasons']
    return dict(fits=fits,geometry=geometry,support=support,reasons=reasons,
        accepted=all(f['reliable'] for f in fits.values()) and geometry['geometry_accepted'] and not reasons)

def temporal_consistency(previous,fits,height,config=TrackingConfig()):
    """Check curve positions and shape at multiple rows, never coefficient deltas."""
    rows=np.linspace(0,height-1,config.temporal_samples);reasons=[];metrics={}
    for side in ('left','right'):
        if not fits[side]['reliable']:
            reasons.append(side+': no current reliable polynomial');continue
        delta=np.polyval(fits[side]['coefficients'],rows)-np.polyval(previous[side],rows)
        position=float(np.max(abs(delta)))
        shape=float(np.ptp(delta))
        metrics[side]=dict(rows=rows.tolist(),delta_px=delta.tolist(),max_position_change_px=position,shape_change_px=shape)
        if position>config.max_position_change_px:reasons.append(side+': abrupt curve position change')
        if shape>config.max_shape_change_px:reasons.append(side+': abrupt curve shape change')
    return dict(accepted=not reasons,reasons=reasons,metrics=metrics,
        note='Checks include extrapolated rows conservatively; shape is the range of positional changes')

class LaneTracker:
    """Keep only the last accepted current-frame pair; discard it on loss."""
    def __init__(self,config=TrackingConfig()):
        self.config=config;self.reset()

    def reset(self):
        self.previous=None;self.frame_index=0;self.state='INITIALIZATION'

    def update(self,binary):
        validate_binary(binary)
        previous=self.previous;attempts=[];chosen=None;search=None
        if previous is not None:
            search=search_around_polynomials(binary,previous,self.config)
            current=evaluate_search(search,binary.shape,self.config)
            temporal=temporal_consistency(previous,current['fits'],binary.shape[0],self.config)
            current['reasons']+=temporal['reasons'];current['accepted'] &= temporal['accepted']
            attempts.append(dict(method='previous_polynomial',accepted=current['accepted'],reasons=current['reasons'].copy(),temporal=temporal))
            if current['accepted']:chosen=current
        if chosen is None:
            search=sliding_window_search(binary)
            current=evaluate_search(search,binary.shape,self.config)
            temporal=temporal_consistency(previous,current['fits'],binary.shape[0],self.config) if previous is not None else None
            if temporal is not None:
                current['reasons']+=temporal['reasons'];current['accepted'] &= temporal['accepted']
            attempts.append(dict(method='sliding_windows',accepted=current['accepted'],reasons=current['reasons'].copy(),temporal=temporal))
            chosen=current
            success_state='INITIALIZATION' if self.frame_index==0 else 'REINITIALIZATION'
        else:success_state='TRACKING'
        accepted=bool(chosen['accepted'])
        self.state=success_state if accepted else 'LOST'
        self.previous={s:chosen['fits'][s]['coefficients'].copy() for s in ('left','right')} if accepted else None
        self.frame_index+=1
        # Never render a pair rejected by additional spatial or temporal checks.
        if not accepted:
            chosen['geometry']['geometry_accepted']=False;chosen['geometry']['offset_m']=None
            chosen['geometry']['reasons']=list(dict.fromkeys(chosen['reasons']))
        return dict(**chosen,search=search,state=self.state,attempts=attempts,
            reinitialization_attempt=previous is not None and len(attempts)==2,
            full_search=attempts[-1]['method']=='sliding_windows')

def summarize(result):
    """Keep numerical diagnostics without pixel arrays or full-height curves."""
    return dict(state=result['state'],lane_rendered=result['accepted'],attempts=result['attempts'],
        reinitialization_attempt=result['reinitialization_attempt'],full_search=result['full_search'],
        initialization=result['search']['initialization'],geometry=result['geometry'],
        warnings=uncertainty_labels(result['geometry']),support=result['support'],
        fits={s:{k:v for k,v in f.items() if k not in ('x','y','supported_rows','extrapolated_rows','interpolated_rows')} for s,f in result['fits'].items()})

def process_frame(frame,tracker,context=None,return_details=False):
    """Process a calibrated BGR frame using shared components and tracker state."""
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.shape!=(720,1280,3):
        raise ValueError('Expected calibrated 1280x720 uint8 BGR frame')
    calibration,matrix,inverse=make_context() if context is None else context
    prepared=undistort_image(frame,calibration)
    binary=warp_binary(binarize_image(prepared),matrix)
    result=tracker.update(binary)
    output,rendered=render_overlay(prepared,result['fits'],result['geometry'],inverse)
    assert rendered==result['accepted']
    cv2.putText(output,'EXPERIMENT '+result['state'],(12,output.shape[0]-16),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    diagnostics=summarize(result)
    if return_details:return output,diagnostics,dict(binary=binary,prepared=prepared,result=result)
    return output,diagnostics
