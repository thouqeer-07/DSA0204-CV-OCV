"""
tracker.py
Multi-Object Tracking (MOT) with Occlusion Handling
(Report Sections 5.4, 5.5, 6.3, 7.5)

Tracking-by-detection, SORT/DeepSORT-inspired:
  - Kalman filter per track: state [cx, cy, w, h, vx, vy, vw]  (~ constant velocity)
  - IoU-based data association solved with the Hungarian algorithm
  - Occlusion handling: unmatched tracks are "coasted" (Kalman-predicted only,
    no correction) for up to MAX_COAST_FRAMES; if re-matched via IoU/appearance
    similarity before the limit, the same ID is retained ("REACQUIRED").
    Otherwise the track is terminated ("LOST") once the limit is exceeded.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from detector import Detection


def iou(bb1, bb2) -> float:
    xx1, yy1 = max(bb1[0], bb2[0]), max(bb1[1], bb2[1])
    xx2, yy2 = min(bb1[2], bb2[2]), min(bb1[3], bb2[3])
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])
    return inter / (area1 + area2 - inter + 1e-9)


def bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([[(x1 + x2) / 2.0], [(y1 + y2) / 2.0], [x2 - x1], [y2 - y1]])


def x_to_bbox(x):
    cx, cy, w, h = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def make_kalman_filter(bbox) -> KalmanFilter:
    """State: [cx, cy, w, h, vcx, vcy, vw]  (7-dim, as in report Section 7.5)."""
    kf = KalmanFilter(dim_x=7, dim_z=4)
    dt = 1.0
    kf.F = np.array([
        [1, 0, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ])
    kf.R[2:, 2:] *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    z = bbox_to_z(bbox)
    kf.x[:4] = z
    return kf


@dataclass
class Track:
    id: int
    cls: str
    kf: KalmanFilter
    embedding: np.ndarray = field(default_factory=lambda: np.random.rand(32))
    missed: int = 0
    status: str = "ACTIVE"          # ACTIVE | OCCLUDED | REACQUIRED | LOST
    history: list = field(default_factory=list)

    def predict(self):
        self.kf.predict()
        return self.get_bbox()

    def update(self, bbox, embedding=None):
        self.kf.update(bbox_to_z(bbox))
        self.missed = 0
        self.status = "ACTIVE"
        if embedding is not None:
            # Exponential moving average of the appearance descriptor
            self.embedding = 0.9 * self.embedding + 0.1 * embedding
        self.history.append(self.get_bbox())

    def get_bbox(self):
        return x_to_bbox(self.kf.x)

    def velocity(self):
        return float(self.kf.x[4, 0]), float(self.kf.x[5, 0])


class MultiObjectTracker:
    """SORT/DeepSORT-style tracker with explicit occlusion (coasting) handling."""

    _next_id = 1

    def __init__(
        self,
        max_coast_frames: int = 20,
        iou_threshold: float = 0.3,
        reid_sim_threshold: float = 0.7,
    ):
        self.tracks: list[Track] = []
        self.max_coast_frames = max_coast_frames
        self.iou_threshold = iou_threshold
        self.reid_sim_threshold = reid_sim_threshold

    def _new_id(self) -> int:
        MultiObjectTracker._next_id += 1
        return MultiObjectTracker._next_id - 1

    @staticmethod
    def _fake_embedding(bbox, frame_shape) -> np.ndarray:
        """Placeholder appearance descriptor (a real system uses a Re-ID CNN)."""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        feat = np.array([
            (x1 + x2) / (2 * w), (y1 + y2) / (2 * h),
            (x2 - x1) / w, (y2 - y1) / h,
        ])
        return np.pad(feat, (0, 28), constant_values=0.0)

    def step(self, detections: list[Detection], frame_shape=(480, 640)) -> list[Track]:
        # 1. Predict every existing track forward (Kalman predict)
        for t in self.tracks:
            t.predict()

        # 2. Associate detections with tracks via IoU cost + Hungarian algorithm
        n_tracks, n_dets = len(self.tracks), len(detections)
        matched_tracks, matched_dets = set(), set()

        if n_tracks and n_dets:
            cost = np.ones((n_tracks, n_dets))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - iou(t.get_bbox(), d.bbox)
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < (1.0 - self.iou_threshold):
                    emb = self._fake_embedding(detections[c].bbox, frame_shape)
                    self.tracks[r].update(detections[c].bbox, emb)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # 3. Occlusion handling for unmatched tracks (coast / re-ID / terminate)
        for i, t in enumerate(self.tracks):
            if i in matched_tracks:
                continue
            t.missed += 1
            reacquired = False
            if t.missed <= self.max_coast_frames:
                # Try Re-ID against remaining unmatched detections
                for j, d in enumerate(detections):
                    if j in matched_dets:
                        continue
                    emb = self._fake_embedding(d.bbox, frame_shape)
                    sim = 1.0 - np.linalg.norm(t.embedding - emb) / (
                        np.linalg.norm(t.embedding) + np.linalg.norm(emb) + 1e-9
                    )
                    if sim > self.reid_sim_threshold and iou(t.get_bbox(), d.bbox) > 0.1:
                        t.update(d.bbox, emb)
                        t.status = "REACQUIRED"
                        matched_dets.add(j)
                        reacquired = True
                        break
                if not reacquired:
                    t.status = "OCCLUDED"  # Kalman-predicted position only, still reported
            else:
                t.status = "LOST"

        self.tracks = [t for t in self.tracks if t.status != "LOST"]

        # 4. Spawn new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j in matched_dets:
                continue
            kf = make_kalman_filter(d.bbox)
            emb = self._fake_embedding(d.bbox, frame_shape)
            self.tracks.append(Track(id=self._new_id(), cls=d.cls, kf=kf, embedding=emb))

        return self.tracks


if __name__ == "__main__":
    mot = MultiObjectTracker(max_coast_frames=5)
    seq = [
        [Detection((100, 100, 150, 200), "pedestrian", 0.9)],
        [Detection((105, 100, 155, 200), "pedestrian", 0.9)],
        [],  # occluded frame
        [],  # still occluded
        [Detection((115, 100, 165, 200), "pedestrian", 0.9)],  # re-appears
    ]
    for frame_idx, dets in enumerate(seq):
        tracks = mot.step(dets)
        print(f"frame {frame_idx}: " + ", ".join(f"id={t.id}:{t.status}" for t in tracks))
