"""
Telemetry: §2 Per-Zone Byte Measurement
=========================================
Measures JPEG-compressed byte size of each zone independently.
Quantifies the compression benefit of zone 2 downsampling.
"""

import cv2


def measure_zone_bytes(frame, ax, ay, aw, ah,
                       rx, ry, rw, rh, jpeg_quality=85):
    """
    Encode each zone crop independently to count compressed bytes.
    This is measurement only: not used for actual output encoding.

    Returns:
        (z1_bytes, z2_bytes, z3_bytes_per_200px)
    """
    def jpeg_size(crop):
        if crop.size == 0:
            return 0
        ok, buf = cv2.imencode('.jpg', crop,
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return len(buf) if ok else 0

    z1_bytes = jpeg_size(frame[ay:ay+ah, ax:ax+aw])
    z2_bytes = jpeg_size(frame[ry:ry+rh, rx:rx+rw])

    # Background sample: 200x200 block from top-left corner
    fh, fw = frame.shape[:2]
    sample_h = min(200, fh)
    sample_w = min(200, fw)
    bg_sample = frame[0:sample_h, 0:sample_w]
    z3_bytes_per_200px = jpeg_size(bg_sample)

    print(
        f"[ZONES] Z1={z1_bytes}B (full-res ROI) | "
        f"Z2={z2_bytes}B (50% ring) | "
        f"Z3~{z3_bytes_per_200px}B/200px (black bg) | "
        f"ratio Z1/Z2={z1_bytes / max(z2_bytes, 1):.2f}"
    )
    return z1_bytes, z2_bytes, z3_bytes_per_200px
