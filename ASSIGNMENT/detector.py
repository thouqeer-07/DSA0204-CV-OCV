"""
detector.py
Object & Pedestrian Detection Module (Report Section 5.2 / 7.3)

Wraps a single-stage CNN detector (YOLO-style, e.g. Ultralytics YOLOv8) with
a favourable speed/accuracy trade-off for the real-time latency budget
(NFR1). Falls back to a lightweight synthetic/motion-blob detector when no
trained weights or the `ultralytics` package are available, so the rest of
the pipeline (tracking, occlusion handling, risk assessment) can still be
demonstrated end-to-end.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

CLASSES = ["pedestrian", "cyclist", "car", "truck_bus", "traffic_sign", "traffic_light"]


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    cls: str
    conf: float


def nms(detections: list[Detection], iou_thresh: float = 0.45) -> list[Detection]:
    """Non-Maximum Suppression to remove duplicate/overlapping boxes."""
    if not detections:
        return []
    boxes = np.array([d.bbox for d in detections], dtype=np.float64)
    scores = np.array([d.conf for d in detections], dtype=np.float64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return [detections[i] for i in keep]


class Detector:
    """
    Thin wrapper around a YOLO-style detector.

    Usage:
        det = Detector(weights="yolov8m_roadsafety.pt", conf_thresh=0.35)
        detections = det.detect(frame)
    """

    def __init__(self, weights: str = "yolov8m_roadsafety.pt", conf_thresh: float = 0.35):
        self.conf_thresh = conf_thresh
        self.classes = CLASSES
        self.model = None
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(weights)
        except Exception:
            # No ultralytics package / no trained weights available in this
            # environment -> use the synthetic fallback detector below.
            self.model = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_fallback(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            detections.append(Detection((x1, y1, x2, y2), cls_name, conf))
        return nms(detections)

    def _detect_fallback(self, frame: np.ndarray) -> list[Detection]:
        """
        Lightweight stand-in detector for demo/testing without trained weights:
        finds salient rectangular blobs via contour detection on edges and
        assigns a pseudo class/confidence by simple heuristics (aspect ratio,
        size). This lets the tracking/occlusion/risk stages run on arbitrary
        video without a trained CNN.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h_img, w_img = frame.shape[:2]
        min_area = 0.0015 * h_img * w_img
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area:
                continue
            aspect = h / max(w, 1)
            if aspect > 1.6:
                cls_name, conf = "pedestrian", 0.55
            elif 0.5 <= aspect <= 1.6 and area > 3 * min_area:
                cls_name, conf = "car", 0.6
            else:
                cls_name, conf = "traffic_sign", 0.4
            detections.append(Detection((x, y, x + w, y + h), cls_name, conf))
        return nms(detections)


if __name__ == "__main__":
    det = Detector()
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets = det.detect(dummy)
    print(f"Fallback detector found {len(dets)} candidate detections on random noise frame.")
