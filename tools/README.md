[↑ Back to Repo Root](../../README.md)

---

# Tools

Utility scripts for setup, verification, and benchmarking. These are **not part of the pipeline** but are essential for working with it.

| Tool | Folder | Purpose |
|------|--------|---------|
| [`preflight/`](preflight/) | `preflight.py` | Pre-run hardware check: verifies all dependencies before starting a pipeline |
| [`benchmark/`](benchmark/) | `realBenchmark.py` | Hardware performance measurement and automated pipeline comparison |

## Recommended Workflow

```
1. Run preflight.py          → confirm hardware is ready
2. Run realBenchmark.py      → measure MJPEG vs H.264 bandwidth
```
