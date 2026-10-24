# Final presentation validation

Independent-frame pipeline with the approved display. Camera identity is an adopted project assumption, not independently verified. Metric values are approximate.

| Video | Resolution | FPS | Decoded frames | Lane rendered | Unavailable |
|---|---|---:|---:|---:|---:|
| project_video01 | 960x540 | 25 | 221 | 221 | 0 |
| project_video02 | 960x540 | 25 | 681 | 681 | 0 |
| project_video03 | 1280x720 | 25 | 1260 | 1243 | 17 |

All 2162 frame diagnostics match the previous final results exactly, excluding processing time. All previous protected artifacts remain unchanged. 53 main tests and 36 experiment tests passed.

All 17 unavailable frames were rechecked: no road overlay or boundary curves below the information panel. MP4 files were fully decoded again to verify resolution, FPS and frame count.

## Documentation candidates

PNG images in docs_candidates/ are decoded from the new MP4 files. They demonstrate the presentation, not independently verified detection accuracy.
- project_video01_000087_960_nearly_straight.png: project_video01, zero-based frame 87.
- project_video02_000103_960_centered.png: project_video02, zero-based frame 103.
- project_video03_000547_1280_extrapolation.png: project_video03, zero-based frame 547.
- project_video03_000997_1280_curvature.png: project_video03, zero-based frame 997.
- project_video03_000500_1280_unavailable.png: project_video03, zero-based frame 500.
