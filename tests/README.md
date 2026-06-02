# Unit Tests

This directory contains automated unit tests to ensure the core algorithms of the pipeline (zone masking, adaptive ROI, and tracking) function correctly.

## Files Explained

### `test_zone_mask.py`
This is the main test suite. It verifies that:
- The 3-zone geometric calculations never go out of frame bounds.
- Zone 1 (full resolution) pixels are perfectly preserved.
- Zone 3 (background) pixels are rendered as pure black.
- The multi-target painter's algorithm correctly handles overlapping zones.
- The `adaptive_pad` function applies more padding in the direction of motion.
- The `CentroidTracker` calculates velocity correctly.

**Important Note:** The original module files on the ZCU104 board use a specific PetaLinux Python 3.7 syntax that can cause `SyntaxError` on standard Python versions. To allow these tests to run anywhere (like GitHub Actions CI or your local Windows/Mac laptop) without needing an FPGA board, `test_zone_mask.py` uses pure-Python reference implementations of the core functions. They behave identically to the hardware versions.

### `__init__.py`
An empty file that tells Python to treat the `tests/` directory as a package. This is required for test runners like `pytest` to correctly discover and execute the tests in this folder.

## Running the Tests

You can run these tests on any computer without needing the ZCU104 board.

```bash
# Install testing dependencies
pip install pytest opencv-python numpy

# Run the tests from the root of the repository
pytest tests/ -v
```
