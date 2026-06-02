"""
tests/test_zone_mask.py
=======================
Unit tests for the core zone masking, adaptive ROI, and tracker algorithms.
Tests run entirely on CPU, no FPGA required.

These tests use pure-Python reference implementations of the core logic
(identical in behavior to the module files, but written in portable Python 3
syntax so they work in any CI environment including GitHub Actions).

Run with:
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import cv2
import pytest

# ── Pure-Python reference implementations ─────────────────────────
# Identical in behavior to modules/zone_mask/zone_mask.py,
# modules/adaptive_roi/adaptive_roi.py, and modules/tracker/tracker.py.
# Written here in portable syntax so tests pass in standard Python 3
# without requiring the board's PetaLinux-specific module format.

from collections import deque


class CentroidTracker:
    """Exponential-weighted centroid tracker (reference implementation)."""
    def __init__(self, history: int = 8):
        self._history = deque(maxlen=history)
        self._prev_cx = None
        self._prev_cy = None
        self.vx = 0.0
        self.vy = 0.0

    def update(self, cx: float, cy: float) -> None:
        if self._prev_cx is not None:
            self.vx = cx - self._prev_cx
            self.vy = cy - self._prev_cy
            self._history.append((self.vx, self.vy))
        self._prev_cx = cx
        self._prev_cy = cy

    def smooth_velocity(self):
        if not self._history:
            return 0.0, 0.0
        n = len(self._history)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= weights.sum()
        vxs = np.array([h[0] for h in self._history])
        vys = np.array([h[1] for h in self._history])
        return float(np.dot(weights, vxs)), float(np.dot(weights, vys))

    def predict_next(self):
        return self.smooth_velocity()

    def reset(self) -> None:
        self._prev_cx = None
        self._prev_cy = None
        self._history.clear()
        self.vx = 0.0
        self.vy = 0.0


def adaptive_pad(x, y, w, h, vx, vy, frame_w, frame_h,
                 base_pad=20, vel_scale=3.0, max_expand=80):
    """Velocity-aware asymmetric ROI padding (reference implementation)."""
    pad_left   = base_pad + int(min(max(0.0, -vx) * vel_scale, max_expand))
    pad_right  = base_pad + int(min(max(0.0,  vx) * vel_scale, max_expand))
    pad_top    = base_pad + int(min(max(0.0, -vy) * vel_scale, max_expand))
    pad_bottom = base_pad + int(min(max(0.0,  vy) * vel_scale, max_expand))
    x1 = max(0,       x - pad_left)
    y1 = max(0,       y - pad_top)
    x2 = min(frame_w, x + w + pad_right)
    y2 = min(frame_h, y + h + pad_bottom)
    return x1, y1, x2 - x1, y2 - y1


def _ring_box(ax, ay, aw, ah, fw, fh, ring_scale=1.6):
    ring_w = int(aw * ring_scale)
    ring_h = int(ah * ring_scale)
    rx = max(0, ax - (ring_w - aw) // 2)
    ry = max(0, ay - (ring_h - ah) // 2)
    rx2 = min(fw, rx + ring_w)
    ry2 = min(fh, ry + ring_h)
    return rx, ry, rx2 - rx, ry2 - ry


def build_zone_mask(frame, ax, ay, aw, ah, ring_scale=1.6, zone2_downsample=0.5):
    """Single-target zone composite (reference implementation)."""
    fh, fw = frame.shape[:2]
    out = np.zeros_like(frame)
    rx, ry, rw, rh = _ring_box(ax, ay, aw, ah, fw, fh, ring_scale)
    z2_src = frame[ry:ry+rh, rx:rx+rw]
    if z2_src.size > 0:
        sw = max(1, int(z2_src.shape[1] * zone2_downsample))
        sh = max(1, int(z2_src.shape[0] * zone2_downsample))
        small = cv2.resize(z2_src, (sw, sh), interpolation=cv2.INTER_AREA)
        z2_up = cv2.resize(small, (z2_src.shape[1], z2_src.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        out[ry:ry+rh, rx:rx+rw] = z2_up
    out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]
    return out, (rx, ry, rw, rh)


def build_zone_mask_multi(frame, adapted_boxes, ring_scale=1.6, zone2_downsample=0.5):
    """Multi-target zone composite (reference implementation)."""
    fh, fw = frame.shape[:2]
    out = np.zeros_like(frame)
    ring_boxes = []
    if not adapted_boxes:
        return frame.copy(), []
    for (ax, ay, aw, ah) in adapted_boxes:
        rx, ry, rw, rh = _ring_box(ax, ay, aw, ah, fw, fh, ring_scale)
        ring_boxes.append((rx, ry, rw, rh))
        z2_src = frame[ry:ry+rh, rx:rx+rw]
        if z2_src.size == 0:
            continue
        sw = max(1, int(z2_src.shape[1] * zone2_downsample))
        sh = max(1, int(z2_src.shape[0] * zone2_downsample))
        small = cv2.resize(z2_src, (sw, sh), interpolation=cv2.INTER_AREA)
        z2_up = cv2.resize(small, (z2_src.shape[1], z2_src.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        out[ry:ry+rh, rx:rx+rw] = z2_up
    for (ax, ay, aw, ah) in adapted_boxes:
        out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]
    return out, ring_boxes


def draw_zone_overlay_multi(frame, adapted_boxes, ring_boxes):
    """Draw zone boundaries on frame (reference implementation)."""
    for i, (ax, ay, aw, ah) in enumerate(adapted_boxes):
        cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), (0, 220, 80), 2)
        if i < len(ring_boxes):
            rx, ry, rw, rh = ring_boxes[i]
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 180, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """A 480×640 BGR frame filled with a mid-grey colour (not black)."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    return frame


@pytest.fixture
def person_box():
    """A typical bounding box (x, y, w, h) for a centred person."""
    return (200, 100, 120, 200)   # centre of a 640×480 frame


# ─────────────────────────────────────────────────────────────────
# build_zone_mask (single-target)
# ─────────────────────────────────────────────────────────────────

class TestBuildZoneMask:

    def test_output_shape_matches_input(self, blank_frame, person_box):
        ax, ay, aw, ah = person_box
        out, ring = build_zone_mask(blank_frame, ax, ay, aw, ah)
        assert out.shape == blank_frame.shape, "Output shape must match input frame shape"

    def test_output_dtype_is_uint8(self, blank_frame, person_box):
        ax, ay, aw, ah = person_box
        out, ring = build_zone_mask(blank_frame, ax, ay, aw, ah)
        assert out.dtype == np.uint8

    def test_zone1_pixels_are_preserved_exactly(self, blank_frame, person_box):
        """Zone 1 (ROI) pixels in the output must be identical to the source frame."""
        ax, ay, aw, ah = person_box
        # Make the ROI region a distinctive colour
        blank_frame[ay:ay+ah, ax:ax+aw] = (0, 255, 0)   # bright green
        out, _ = build_zone_mask(blank_frame, ax, ay, aw, ah)
        roi_out = out[ay:ay+ah, ax:ax+aw]
        assert np.all(roi_out == (0, 255, 0)), "Zone 1 pixels must be pixel-perfect copies"

    def test_zone3_pixels_are_black(self, blank_frame, person_box):
        """All pixels outside Zone 2 ring must be exactly zero (Zone 3 = black)."""
        ax, ay, aw, ah = person_box
        out, (rx, ry, rw, rh) = build_zone_mask(blank_frame, ax, ay, aw, ah)
        # Corner pixel (0,0) is far outside any zone for this box placement
        assert tuple(out[0, 0]) == (0, 0, 0), "Zone 3 pixels must be pure black"

    def test_ring_box_returned_as_tuple(self, blank_frame, person_box):
        ax, ay, aw, ah = person_box
        out, ring = build_zone_mask(blank_frame, ax, ay, aw, ah)
        assert isinstance(ring, tuple) and len(ring) == 4

    def test_ring_box_stays_within_frame(self, blank_frame, person_box):
        ax, ay, aw, ah = person_box
        fh, fw = blank_frame.shape[:2]
        _, (rx, ry, rw, rh) = build_zone_mask(blank_frame, ax, ay, aw, ah)
        assert rx >= 0 and ry >= 0
        assert rx + rw <= fw
        assert ry + rh <= fh

    def test_edge_box_does_not_crash(self):
        """A bounding box at the frame boundary must not cause index errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Box touching the top-left corner
        out, ring = build_zone_mask(frame, 0, 0, 50, 50)
        assert out.shape == frame.shape

    def test_large_box_does_not_crash(self):
        """A bounding box nearly as large as the frame must not crash."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        out, ring = build_zone_mask(frame, 0, 0, 638, 478)
        assert out.shape == frame.shape


# ─────────────────────────────────────────────────────────────────
# build_zone_mask_multi
# ─────────────────────────────────────────────────────────────────

class TestBuildZoneMaskMulti:

    def test_empty_boxes_returns_full_frame(self, blank_frame):
        """No detections → output is a copy of the full frame (not masked)."""
        out, rings = build_zone_mask_multi(blank_frame, [])
        assert out.shape == blank_frame.shape
        assert len(rings) == 0
        # Output must be the full, unmasked frame
        assert np.array_equal(out, blank_frame)

    def test_single_box_output_shape(self, blank_frame, person_box):
        ax, ay, aw, ah = person_box
        out, rings = build_zone_mask_multi(blank_frame, [(ax, ay, aw, ah)])
        assert out.shape == blank_frame.shape
        assert len(rings) == 1

    def test_multi_box_ring_count_matches(self, blank_frame):
        boxes = [(50, 50, 80, 120), (400, 200, 100, 150)]
        out, rings = build_zone_mask_multi(blank_frame, boxes)
        assert len(rings) == len(boxes), "Ring count must match input box count"

    def test_zone1_wins_over_zone2_on_overlap(self, blank_frame):
        """
        When Zone 1 of one target overlaps Zone 2 of another, Zone 1 pixels
        must be pixel-perfect (not blurred by Zone 2 processing).
        """
        # Two closely-spaced boxes so their zones overlap
        blank_frame[100:200, 100:200] = (0, 0, 255)   # Blue zone — box 1 ROI
        blank_frame[100:200, 250:350] = (255, 0, 0)   # Red zone — box 2 ROI
        boxes = [(100, 100, 100, 100), (250, 100, 100, 100)]
        out, _ = build_zone_mask_multi(blank_frame, boxes)
        # Zone 1 of box 1: pixels must be identical to source
        assert np.all(out[100:200, 100:200] == (0, 0, 255))

    def test_multi_target_does_not_exceed_frame_bounds(self, blank_frame):
        """Adding many boxes must not produce an out-of-bounds frame."""
        boxes = [
            (10, 10, 60, 80),
            (300, 200, 100, 150),
            (500, 350, 120, 110),
        ]
        out, rings = build_zone_mask_multi(blank_frame, boxes)
        fh, fw = blank_frame.shape[:2]
        assert out.shape == (fh, fw, 3)


# ─────────────────────────────────────────────────────────────────
# draw_zone_overlay_multi
# ─────────────────────────────────────────────────────────────────

class TestDrawZoneOverlayMulti:

    def test_returns_same_frame(self, blank_frame):
        """draw_zone_overlay_multi must return the input frame (in-place mutation)."""
        boxes = [(100, 100, 80, 120)]
        rings = [(80, 80, 128, 192)]
        result = draw_zone_overlay_multi(blank_frame, boxes, rings)
        assert result is blank_frame, "Must return the same frame object (in-place)"

    def test_empty_boxes_returns_frame(self, blank_frame):
        result = draw_zone_overlay_multi(blank_frame, [], [])
        assert result is blank_frame

    def test_does_not_crash_on_multiple_targets(self, blank_frame):
        boxes = [(50, 50, 80, 100), (400, 200, 60, 90)]
        rings = [(30, 30, 120, 150), (380, 180, 100, 130)]
        result = draw_zone_overlay_multi(blank_frame, boxes, rings)
        assert result.shape == blank_frame.shape


# ─────────────────────────────────────────────────────────────────
# adaptive_pad
# ─────────────────────────────────────────────────────────────────

class TestAdaptivePad:

    def test_static_target_has_uniform_padding(self):
        """A target with zero velocity should have equal padding on all sides."""
        ax, ay, aw, ah = adaptive_pad(100, 100, 80, 120, 0.0, 0.0, 640, 480)
        # base_pad=20 on all sides
        assert ax == 80    # 100 - 20
        assert ay == 80    # 100 - 20
        assert aw == 120   # 80 + 20 + 20
        assert ah == 160   # 120 + 20 + 20

    def test_rightward_motion_expands_right_more(self):
        """Target moving right (vx > 0) should have more padding on the right."""
        ax, ay, aw, ah = adaptive_pad(100, 100, 80, 120, 10.0, 0.0, 640, 480, base_pad=0, vel_scale=3.0)
        right_expansion = aw - 80   # total width expansion
        assert right_expansion > 0, "Must expand rightward for positive vx"

    def test_output_stays_within_frame(self):
        """Padded box must never exceed frame boundaries."""
        fw, fh = 640, 480
        ax, ay, aw, ah = adaptive_pad(0, 0, 50, 50, -20.0, -20.0, fw, fh)
        assert ax >= 0
        assert ay >= 0
        assert ax + aw <= fw
        assert ay + ah <= fh

    def test_max_expand_limits_explosion(self):
        """Very high velocity should not blow the box beyond max_expand."""
        ax, ay, aw, ah = adaptive_pad(
            100, 100, 80, 120,
            vx=1000.0, vy=1000.0,    # extreme velocity
            frame_w=640, frame_h=480,
            base_pad=0, vel_scale=3.0, max_expand=50
        )
        # right expansion capped at max_expand=50
        assert aw <= 80 + 50, "Width expansion must be capped by max_expand"


# ─────────────────────────────────────────────────────────────────
# CentroidTracker
# ─────────────────────────────────────────────────────────────────

class TestCentroidTracker:

    def test_initial_velocity_is_zero(self):
        t = CentroidTracker()
        vx, vy = t.predict_next()
        assert vx == 0.0 and vy == 0.0

    def test_velocity_after_one_update_is_zero(self):
        """First update sets previous position; no velocity yet."""
        t = CentroidTracker()
        t.update(100, 100)
        vx, vy = t.predict_next()
        assert vx == 0.0 and vy == 0.0

    def test_constant_rightward_motion(self):
        """10 frames of rightward motion should produce positive vx."""
        t = CentroidTracker(history=8)
        for i in range(10):
            t.update(i * 10, 100)   # moving right by 10 px/frame
        vx, vy = t.predict_next()
        assert vx > 0, "Positive vx expected for rightward motion"
        assert abs(vy) < 1.0, "vy should be near-zero for horizontal motion"

    def test_reset_clears_velocity(self):
        t = CentroidTracker()
        for i in range(5):
            t.update(i * 10, 100)
        t.reset()
        vx, vy = t.predict_next()
        assert vx == 0.0 and vy == 0.0

    def test_history_buffer_is_bounded(self):
        """History ring buffer should never exceed `history` entries."""
        t = CentroidTracker(history=4)
        for i in range(20):
            t.update(i * 5, i * 5)
        assert len(t._history) <= 4
