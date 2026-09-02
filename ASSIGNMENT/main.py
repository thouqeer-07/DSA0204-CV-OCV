"""
main.py
RoadSafetyPipeline — end-to-end orchestration of the six-stage pipeline
described in the report (Section 5 / 6.2 pseudocode):

  1. Camera calibration & pre-processing (calibration.py)
  2. Object & pedestrian detection        (detector.py)
  3. Road-sign & lane-marking recognition (lane_detection.py)
  4. Multi-object tracking                (tracker.py)
  5. Occlusion handling                   (integrated into tracker.py)
  6. Risk assessment & alerting           (risk.py)

Usage:
    python main.py --video sample_data/demo.mp4 --output output/annotated.mp4
    python main.py                      # generates + runs on a synthetic demo video
"""
from __future__ import annotations
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

from calibration import synthetic_calibration
from detector import Detector
from lane_detection import detect_lane_markings, fit_lane_lines, draw_lanes
from tracker import MultiObjectTracker
from risk import assess_track, px_velocity_to_metric


CLASS_COLORS = {
    "pedestrian": (30, 30, 220),
    "cyclist": (30, 180, 220),
    "car": (200, 60, 30),
    "truck_bus": (90, 90, 90),
    "traffic_sign": (0, 200, 255),
    "traffic_light": (0, 255, 0),
}


def draw_track(frame, track, risk=None):
    x1, y1, x2, y2 = [int(v) for v in track.get_bbox()]
    color = CLASS_COLORS.get(track.cls, (255, 255, 255))
    style = cv2.LINE_AA
    thickness = 2 if track.status == "ACTIVE" else 1
    if track.status in ("OCCLUDED",):
        # dashed-effect approximation: draw a thinner, lighter box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, style)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, style)

    label = f"{track.cls}#{track.id} {track.status}"
    if risk is not None:
        label += f" TTC={risk.ttc:.1f}s R={risk.risk:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, style)
    if risk is not None and risk.alert:
        cv2.putText(frame, "!!ALERT!!", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, style)


def run_pipeline(video_path: str, output_path: str, log_path: str,
                  max_coast_frames: int = 20, risk_threshold: float = 0.55) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    calibrator = synthetic_calibration((w, h))
    detector = Detector()
    tracker = MultiObjectTracker(max_coast_frames=max_coast_frames)

    log_rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        # 1. Calibration / undistortion
        frame_u = calibrator.undistort(frame)

        # 2. Object & pedestrian detection
        detections = detector.detect(frame_u)

        # 3. Lane / road-marking recognition
        lines = detect_lane_markings(frame_u)
        lane_fit = fit_lane_lines(lines, frame_u.shape)

        # 4 & 5. Tracking + occlusion handling
        tracks = tracker.step(detections, frame_u.shape)

        # 6. Risk assessment & alerting
        out_frame = draw_lanes(frame_u, lane_fit)
        for t in tracks:
            vx, vy = t.velocity()
            closing_speed, lateral_offset = px_velocity_to_metric(vx, vy)
            distance_m = max(1.0, 30.0 - 0.05 * t.get_bbox()[3])  # crude pseudo-distance proxy
            risk = assess_track(t.id, t.cls, distance_m, closing_speed, lateral_offset,
                                 threshold=risk_threshold)
            draw_track(out_frame, t, risk)
            log_rows.append({
                "frame": frame_idx, "track_id": t.id, "class": t.cls,
                "status": t.status, "ttc_s": risk.ttc, "risk": risk.risk,
                "alert": risk.alert,
            })

        latency_ms = (time.time() - t0) * 1000
        cv2.putText(out_frame, f"Frame {frame_idx} | {latency_ms:.1f} ms/frame",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if log_rows:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f"Processed {frame_idx} frames.")
    print(f"Annotated video -> {output_path}")
    print(f"Per-track performance log -> {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Computer-Vision Based Autonomous Road Safety System")
    p.add_argument("--video", default=None, help="Path to input video. If omitted, a synthetic demo video is generated.")
    p.add_argument("--output", default="output/annotated.mp4", help="Path to write annotated output video.")
    p.add_argument("--log", default="output/track_log.csv", help="Path to write per-frame/track CSV log.")
    p.add_argument("--max-coast-frames", type=int, default=20, help="Occlusion coasting limit (Section 5.5).")
    p.add_argument("--risk-threshold", type=float, default=0.55, help="Alert threshold (Section 7.6).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video
    if video_path is None:
        from make_demo_video import generate_demo_video
        os.makedirs("sample_data", exist_ok=True)
        video_path = generate_demo_video("sample_data/demo.mp4")
        print(f"No --video given; generated synthetic demo video at {video_path}")

    run_pipeline(video_path, args.output, args.log,
                 max_coast_frames=args.max_coast_frames,
                 risk_threshold=args.risk_threshold)
