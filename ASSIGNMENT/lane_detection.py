"""
lane_detection.py
Road-Sign and Road-Marking Recognition Module (Report Section 5.3 / 7.4)

Classical CV pipeline for lane/road-marking detection:
grayscale -> Gaussian blur -> Canny edge detection -> region-of-interest
masking -> probabilistic Hough transform.
"""

from __future__ import annotations
import numpy as np
import cv2


def region_of_interest_mask(edges: np.ndarray) -> np.ndarray:
    h, w = edges.shape
    mask = np.zeros_like(edges)
    roi = np.array(
        [[(0, h), (int(w * 0.45), int(h * 0.6)), (int(w * 0.55), int(h * 0.6)), (w, h)]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, roi, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lane_markings(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_thresh: int = 40,
    min_line_length: int = 40,
    max_line_gap: int = 120,
) -> np.ndarray:
    """
    Returns an array of shape (N, 1, 4) of [x1, y1, x2, y2] line segments,
    or an empty array if none found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    masked = region_of_interest_mask(edges)
    lines = cv2.HoughLinesP(
        masked, 1, np.pi / 180, hough_thresh,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    return lines if lines is not None else np.empty((0, 1, 4), dtype=np.int32)


def fit_lane_lines(lines: np.ndarray, frame_shape: tuple[int, int]) -> dict:
    """
    Separates Hough segments into left/right lane candidates by slope sign
    and fits a single representative line to each side (simple linear fit),
    approximating the 'lane geometry estimation' described in Section 4.1.
    """
    h, w = frame_shape[:2]
    left_pts, right_pts = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:  # near-horizontal, likely not a lane edge
            continue
        (left_pts if slope < 0 else right_pts).extend([(x1, y1), (x2, y2)])

    def fit_side(pts):
        if len(pts) < 2:
            return None
        pts = np.array(pts)
        poly = np.polyfit(pts[:, 1], pts[:, 0], deg=1)  # x = m*y + b
        y_bottom, y_top = h, int(h * 0.6)
        x_bottom = int(np.polyval(poly, y_bottom))
        x_top = int(np.polyval(poly, y_top))
        return (x_bottom, y_bottom, x_top, y_top)

    return {"left": fit_side(left_pts), "right": fit_side(right_pts)}


def draw_lanes(frame: np.ndarray, lane_fit: dict) -> np.ndarray:
    out = frame.copy()
    for side, color in (("left", (0, 255, 255)), ("right", (255, 255, 0))):
        line = lane_fit.get(side)
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(out, (x1, y1), (x2, y2), color, 4)
    return out


if __name__ == "__main__":
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(dummy, (100, 480), (300, 300), (255, 255, 255), 4)
    cv2.line(dummy, (600, 480), (400, 300), (255, 255, 255), 4)
    lines = detect_lane_markings(dummy)
    fit = fit_lane_lines(lines, dummy.shape)
    print("Detected line segments:", len(lines))
    print("Fitted lane sides:", fit)
