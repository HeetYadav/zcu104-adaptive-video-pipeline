"""
zone_mask.py — 3-zone quality tiering (Option C), multi-target variant.

Zone 1 (inside adaptive ROI)    : full-resolution, preserved exactly
Zone 2 (proximity ring ~1.6×)   : downsampled to 50%, then upsampled
Zone 3 (everything else)        : pure black (zero-value → near-zero bits)

Supports up to MAX_FACES simultaneous targets; zones from different targets
are merged onto one canvas, with Zone 1 always winning priority.
"""

import cv2
import numpy as np


# ── Zone 2 ring geometry ──────────────────────────────────────────
def _ring_box(ax, ay, aw, ah, fw, fh, ring_scale=1.6):
    """Compute the Zone-2 bounding rect for one adaptive ROI box."""
    ring_w = int(aw * ring_scale)
    ring_h = int(ah * ring_scale)
    rx = max(0,  ax - (ring_w - aw) // 2)
    ry = max(0,  ay - (ring_h - ah) // 2)
    rx2 = min(fw, rx + ring_w)
    ry2 = min(fh, ry + ring_h)
    return rx, ry, rx2 - rx, ry2 - ry


# ── Single-target helpers (kept for unit-test convenience) ────────
def build_zone_mask(frame, ax, ay, aw, ah,
                    ring_scale=1.6, zone2_downsample=0.5):
    """
    Single-target zone composite.
    Returns (composited_frame, (rx, ry, rw, rh)).
    """
    fh, fw = frame.shape[:2]
    out = np.zeros_like(frame)

    rx, ry, rw, rh = _ring_box(ax, ay, aw, ah, fw, fh, ring_scale)

    # Zone 2: downsample the ring crop, paste into output
    z2_src = frame[ry:ry+rh, rx:rx+rw]
    if z2_src.size > 0:
        sw = max(1, int(z2_src.shape[1] * zone2_downsample))
        sh = max(1, int(z2_src.shape[0] * zone2_downsample))
        small  = cv2.resize(z2_src, (sw, sh), interpolation=cv2.INTER_AREA)
        z2_up  = cv2.resize(small,  (z2_src.shape[1], z2_src.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        out[ry:ry+rh, rx:rx+rw] = z2_up

    # Zone 1: paste full-res ROI on top (wins over zone 2)
    out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]

    return out, (rx, ry, rw, rh)


# ── Multi-target variants (used by pipeline_hw.py) ────────────────
def build_zone_mask_multi(frame, adapted_boxes,
                          ring_scale=1.6, zone2_downsample=0.5):
    """
    Multi-target zone composite.

    Parameters
    ----------
    frame          : BGR frame (numpy array)
    adapted_boxes  : list of (ax, ay, aw, ah) from adaptive_pad()

    Returns
    -------
    (composited_frame, ring_boxes)
      ring_boxes is a list of (rx, ry, rw, rh) parallel to adapted_boxes
    """
    fh, fw = frame.shape[:2]
    out = np.zeros_like(frame)
    ring_boxes = []

    if not adapted_boxes:
        # No targets — return full frame unmodified
        return frame.copy(), []

    # Pass 1: paint all Zone-2 rings first (lower priority)
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

    # Pass 2: paint all Zone-1 ROIs on top (highest priority)
    for (ax, ay, aw, ah) in adapted_boxes:
        out[ay:ay+ah, ax:ax+aw] = frame[ay:ay+ah, ax:ax+aw]

    return out, ring_boxes


# ── Overlay drawing ───────────────────────────────────────────────
def _dashed_rect(img, pt1, pt2, color, thickness=1, dash=8, gap=5):
    """Draw a dashed rectangle. OpenCV has no native dashed line."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash + gap):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash + gap):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_zone_overlay(frame, ax, ay, aw, ah, rx, ry, rw, rh):
    """Single-target overlay (convenience wrapper)."""
    return draw_zone_overlay_multi(frame, [(ax, ay, aw, ah)], [(rx, ry, rw, rh)])


def draw_zone_overlay_multi(frame, adapted_boxes, ring_boxes):
    """
    Draw Zone 1 (solid green) and Zone 2 (dashed amber) boundaries
    on the composited frame.  Mutates frame in-place and returns it.
    """
    for i, (ax, ay, aw, ah) in enumerate(adapted_boxes):
        # Zone 1 — solid green
        cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 220, 80), 2)
        cv2.putText(frame, "Z1", (ax + 4, ay + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 80), 1,
                    cv2.LINE_AA)

        if i < len(ring_boxes):
            rx, ry, rw, rh = ring_boxes[i]
            # Zone 2 — dashed amber
            _dashed_rect(frame, (rx, ry), (rx + rw, ry + rh),
                         (0, 180, 255), thickness=1)
            cv2.putText(frame, "Z2", (rx + 4, ry + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1,
                        cv2.LINE_AA)

    return frame