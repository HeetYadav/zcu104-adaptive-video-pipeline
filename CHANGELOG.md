# Changelog

All notable changes to this project are documented here.  
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- `docs/00_prerequisites.md` — FPGA, ZCU104, DPU, INT8 and GStreamer orientation guide for newcomers
- `requirements.txt` — explicit Python dependency list with board vs. laptop split
- `.github/ISSUE_TEMPLATE/` — structured bug report and feature request forms
- `.github/workflows/ci.yml` — GitHub Actions CI for unit tests and syntax checks
- `tests/test_zone_mask.py` — unit tests for zone masking geometry (runs without hardware)
- `pipelines/pipeline_sim/pipeline_sim.py` — CPU-only simulation mode; runs on any laptop
- `pipelines/pipeline_sim/README.md` — simulation pipeline documentation
- "What Could You Build Next?" section in `README.md`

### Changed
- All 9 `docs/*.md` files: Table of Contents converted from single-line to proper multi-line Markdown lists
- `docs/02_hardware_setup.md`: added Vitis AI TRD 2020.2 exact download URL and Etcher flashing instructions
- `README.md`: docs table now includes `00_prerequisites.md`

### Removed
- `yolov8n.onnx` (12.8 MB): file was untracked, unreferenced, and undocumented — removed to keep the repo clean

---

## [1.0.0] — 2026-05-28

### Final Release: Full Hardware Acceleration

**Headline result: 10.5× bandwidth reduction over MJPEG baseline.**

#### Added
- `pipelines/pipeline_hw/pipeline_hw.py` — Final pipeline: DPU + 3-zone ROI + VCU H.264 + MJPEG HTTP
  - VCU `omxh264enc` hardware H.264 encoder in Variable Bitrate mode
  - Parallel MJPEG HTTP stream for smooth VLC visualization
  - Bandwidth telemetry via active pixel area ratio model
- `tools/benchmark/realBenchmark.py` — Automated benchmark: runs both pipelines, collects telemetry, reports comparison
- `tools/preflight/preflight.py` — Hardware preflight checker: verifies all dependencies before pipeline launch
- `assets/diagrams/` — Architecture diagram, zone masking diagram, benchmark chart, terminal screenshot
- `assets/photos/` — Real board hardware photos and VLC stream screenshots
- `CONTRIBUTING.md` — Contribution guide with photo naming conventions and large-file rules
- Full `docs/` suite: 9 numbered documents covering every aspect of the system

#### Fixed
- **Critical: `xir::DataType::UNKNOWN` abort** — INT8 type mismatch when passing `uint8` to the DPU runner
  - Fix: `(rgb.astype(np.int16) - 128).astype(np.int8)` in the detector thread
- **Critical: `std::bad_any_cast` RTTI crash** — C++ ABI symbol namespace collision between `vart.so` and `xir.so`
  - Fix: `sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)` before any Vitis AI import
- **OSError: Address already in use** — TCP port 5000 in `TIME_WAIT` after pipeline kill
  - Fix: `socketserver.TCPServer.allow_reuse_address = True`
- **DPU Resource Lock hang** — `SIGKILL` left `/tmp/vart_device_0` lock file orphaned
  - Fix: `realBenchmark.py` waits for process group to fully terminate between runs
- **Choppy VLC playback (freeze→fast-forward loop)** — MPEG-TS PTS timestamps caused VLC buffering
  - Fix: switched to raw MJPEG multipart HTTP (no container, no timestamps)

---

## [0.1.0] — 2026-05-20

### Baseline: DPU Inference + MJPEG

- `pipelines/pipeline_hw_1/pipeline_hw_1.py` — Baseline pipeline: DPU + 3-zone ROI + MJPEG HTTP stream (no VCU)
  - Proves: DPU runs YOLOv4 reliably in real-time at the edge
  - Proves: 3-zone masking reduces visible frame content proportionally to detected occupancy
  - Bandwidth: ~6,000–8,821 kbps (MJPEG, CPU-encoded)
  - FPS: ~8.6
- `modules/zone_mask/zone_mask.py` — 3-zone masking engine (multi-target painter's algorithm)
- `modules/adaptive_roi/adaptive_roi.py` — Motion-predictive asymmetric ROI padding
- `modules/tracker/tracker.py` — `CentroidTracker` with exponential-weighted velocity smoothing
- `modules/telemetry/` — Bandwidth telemetry helpers

---

[Unreleased]: https://github.com/HeetYadav/zcu104-adaptive-video-pipeline/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/HeetYadav/zcu104-adaptive-video-pipeline/releases/tag/v1.0.0
[0.1.0]: https://github.com/HeetYadav/zcu104-adaptive-video-pipeline/releases/tag/v0.1.0
