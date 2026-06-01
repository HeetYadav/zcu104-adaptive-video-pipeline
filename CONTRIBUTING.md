# Contributing to ZCU104 ROI Bandwidth Management

Thank you for your interest in contributing! This project runs on real FPGA hardware, so contributions are especially valued: reproducibility and documentation accuracy matter enormously.

---

## Table of Contents

- [Reporting a Bug](#reporting-a-bug)- [Submitting a Fix](#submitting-a-fix)- [Adding Hardware Photos](#adding-hardware-photos)- [Large File Rules](#large-file-rules)- [Code Style](#code-style)
---

## Reporting a Bug

Open an Issue and include:

1. **Exact error message**: paste the full terminal output, including the kernel DRM logs if present
2. **Board + OS version**: `cat /etc/os-release` output from the ZCU104
3. **Python version**: `python3 --version`
4. **Which script**: `pipeline_hw.py`, `pipeline_hw_1.py`, or `realBenchmark.py`
5. **Network config**: phone IP, board IP, any custom changes to `PHONE_HOST`

> [!TIP]
> Run `python3 preflight.py` first: it checks every dependency and hardware driver. Include its output in the issue.

---

## Submitting a Fix

1. Fork the repository
2. Create a branch: `git checkout -b fix/your-description`
3. Make your changes
4. Test on real hardware if possible: or describe exactly what you tested in the PR description
5. Open a Pull Request with:
   - **What broke**: link to the issue   - **Root cause**: one sentence explanation   - **What changed**: specific files + lines   - **How you tested**: terminal output showing it works
> [!WARNING]
> Do NOT commit large model files (`.xmodel`, `.weights`, `.tar.gz`). They are in `.gitignore`: see [Large File Rules](#large-file-rules).

---

## Adding Hardware Photos

Hardware photos are the biggest gap in this documentation. They are **very welcome**.

### Naming Convention

| What | Filename | Location |
|------|----------|----------|
| Full setup on desk | `setup_desk_overview.jpg` | `assets/photos/` |
| JTAG/UART cable detail | `setup_uart_connection.jpg` | `assets/photos/` |
| VLC output screenshot | `vlc_stream_output.png` | `assets/photos/` |
| Detection boxes visible | `vlc_detection_zones.png` | `assets/photos/` |
| Benchmark terminal output | `benchmark_terminal.png` | `assets/photos/` |

### Photo Requirements
- **Resolution:** Minimum 1280×720, prefer 1920×1080- **Format:** `.jpg` for board/setup photos, `.png` for screenshots- **File size:** Keep under 5 MB per photo (GitHub renders large images slowly)- **Lighting:** No glare on the board, labels readable
### Where to Place Them in the Docs
Search for `> [!IMPORTANT]` blocks containing `📸 Add photo here`: each one tells you the exact filename it expects. Replace the callout block with a standard Markdown image embed:

```markdown
![Full Hardware Setup on Desk](../assets/photos/setup_desk_overview.jpg)
```

---

## Large File Rules

The `.gitignore` excludes these file types: **never force-add them**:

| Pattern | Why Excluded |
|---------|-------------|
| `*.xmodel` | Compiled DPU model: 10–100 MB, architecture-specific |
| `*.tar.gz` | Model zoo archives: 40+ MB |
| `*.weights` | YOLOv4 weight files: 23+ MB |
| `*.onnx` | ONNX models: 12+ MB |

Add download instructions to `docs/02_hardware_setup.md` instead.

---

## Code Style

- **Python 3.7+** compatible (the ZCU104 Vitis AI image ships Python 3.7)- **No new external dependencies** without updating `docs/02_hardware_setup.md`- **Thread safety:** Any shared state must be protected by a `threading.Lock()`- **Comments:** Explain *why*, not *what*: the code already shows what