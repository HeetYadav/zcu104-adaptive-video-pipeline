"""
adaptive_roi.py — Velocity-aware ROI padding (Option A).
Expands the bounding box asymmetrically in the direction of motion
so the target never exits the ROI between detection frames.
"""


def adaptive_pad(
    x: int, y: int, w: int, h: int,
    vx: float, vy: float,
    frame_w: int, frame_h: int,
    base_pad: int   = 20,
    vel_scale: float = 3.0,
    max_expand: int  = 80,
):
    """
    Parameters
    ----------
    x, y, w, h     : raw detection bounding box (pixels)
    vx, vy          : velocity from CentroidTracker.predict_next()
    frame_w/h       : frame dimensions for clamping
    base_pad        : minimum padding on every side (pixels)
    vel_scale       : extra pixels of padding per pixel/frame of velocity
    max_expand      : hard cap on velocity-driven expansion

    Returns
    -------
    (ax, ay, aw, ah) : expanded box, clamped to frame boundaries
    """
    # Directional padding: expand in the direction of motion
    pad_left   = base_pad + int(min(max(0.0, -vx) * vel_scale, max_expand))
    pad_right  = base_pad + int(min(max(0.0,  vx) * vel_scale, max_expand))
    pad_top    = base_pad + int(min(max(0.0, -vy) * vel_scale, max_expand))
    pad_bottom = base_pad + int(min(max(0.0,  vy) * vel_scale, max_expand))

    x1 = max(0,        x - pad_left)
    y1 = max(0,        y - pad_top)
    x2 = min(frame_w,  x + w + pad_right)
    y2 = min(frame_h,  y + h + pad_bottom)

    return x1, y1, x2 - x1, y2 - y1
