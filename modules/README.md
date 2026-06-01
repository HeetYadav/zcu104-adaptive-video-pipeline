[↑ Back to Repo Root](../../README.md) | [Docs →](../../docs/)

---

# Modules

Reusable algorithm modules shared across all three pipelines. Each module has a single, focused responsibility and no dependencies on other modules.

| Module | File | What it does |
|--------|------|-------------|
| [`zone_mask/`](zone_mask/) | `zone_mask.py` | 3-zone spatial masking — builds the black-background output frame |
| [`adaptive_roi/`](adaptive_roi/) | `adaptive_roi.py` | Motion-predictive bounding box expansion |
| [`tracker/`](tracker/) | `tracker.py` | Centroid tracker with exponential-weighted velocity smoothing |
| [`telemetry/`](telemetry/) | `telemetry.py` | Per-zone JPEG byte measurement for compression analysis |

## Module Dependency Graph

```
pipeline_hw.py
    ├── zone_mask.py      (build_zone_mask_multi, draw_zone_overlay_multi)
    ├── adaptive_roi.py   (adaptive_pad)
    └── tracker.py        (CentroidTracker)

pipeline_hw_1.py  (same as pipeline_hw.py)

pipeline.py (Phase 2)
    ├── zone_mask.py
    ├── adaptive_roi.py
    ├── tracker.py
    └── telemetry.py      (measure_zone_bytes — Phase 2 only)
```

## Import Path

All modules resolve their path automatically. The pipelines add each module directory to `sys.path` using `__file__`:
```python
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _mod in ['zone_mask', 'adaptive_roi', 'tracker', 'telemetry']:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'modules', _mod))
```
