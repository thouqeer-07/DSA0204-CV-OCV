# Computer-Vision Based Autonomous Road Safety System

Implementation companion to the report *"Computer Vision Based Autonomous
Road Safety System: Integrating Camera Calibration, Detection, Recognition,
Multi-Object Tracking and Occlusion Handling"* (Syed Thouqeer Ahmed A,
192324296).

A modular six-stage pipeline:

1. **`calibration.py`** — Camera calibration (Zhang's method, chessboard
   pattern) and per-frame undistortion.
2. **`detector.py`** — Object & pedestrian detection. Wraps an
   Ultralytics YOLO model if installed/weighted; otherwise falls back to a
   lightweight contour-based detector so the rest of the pipeline still
   runs without trained weights.
3. **`lane_detection.py`** — Classical lane/road-marking detection
   (Canny + ROI mask + probabilistic Hough transform) with simple
   left/right lane line fitting.
4. **`tracker.py`** — Multi-object tracking: Kalman filter per track +
   Hungarian-algorithm IoU association (SORT/DeepSORT-style), including
   **occlusion handling** — unmatched tracks are "coasted" via
   motion-model prediction for up to `MAX_COAST_FRAMES` and re-identified
   via an appearance descriptor before being terminated.
5. **`risk.py`** — Time-To-Collision (TTC) and composite risk scoring per
   track, with a configurable alert threshold.
6. **`main.py`** — Orchestrates all stages end-to-end over a video file,
   writes an annotated output video and a per-frame/per-track CSV log.
7. **`make_demo_video.py`** — Generates a short synthetic driving scene
   (car, pedestrian, and a temporary occlusion event) so the pipeline can
   be run without external datasets (the report's target datasets —
   KITTI, BDD100K, GTSRB — are listed in the references).

## Quick start

```bash
pip install -r requirements.txt

# Run on a synthetic demo video (auto-generated):
python main.py

# Run on your own video:
python main.py --video path/to/clip.mp4 --output output/annotated.mp4
```

Outputs:
- `output/annotated.mp4` — video annotated with detection boxes, track
  IDs/status (ACTIVE / OCCLUDED / REACQUIRED), TTC and risk scores, and
  alert flags.
- `output/track_log.csv` — per-frame, per-track log (class, status, TTC,
  risk, alert) usable for the accuracy/latency analysis described in
  Section 10 of the report.

## Using a real trained detector

By default `detector.py` uses a simple fallback detector (contour-based)
so the pipeline runs with zero external dependencies on trained weights.
To use a real YOLOv8 model as described in the report:

```bash
pip install ultralytics
```

```python
from detector import Detector
det = Detector(weights="yolov8m_roadsafety.pt", conf_thresh=0.35)
```

## Calibrating against a real camera

```python
from calibration import CameraCalibrator
cal = CameraCalibrator(pattern_size=(9, 6))
K, D = cal.calibrate("path/to/chessboard_images/*.jpg")
cal.save("calib_params.npz")
```

Then load it in the pipeline instead of `synthetic_calibration()`.

## Notes

- This is a demonstration-grade implementation of the pipeline
  architecture and algorithms described in the report, intended to run
  end-to-end on arbitrary/sample video. Production deployment would
  require: trained detection weights on the target classes, ground-truth
  evaluation data, sensor fusion for extreme weather (per report Section
  13), and hardware-specific latency optimization for the target edge
  device (report NFR1, Section 10.5).
