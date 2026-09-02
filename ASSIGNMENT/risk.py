"""
risk.py
Risk Assessment and Alerting Module (Report Section 5.6 / 7.6)

Computes Time-To-Collision (TTC) and a composite risk score per tracked
object, combining TTC, lateral proximity to the ego-path and object class
(pedestrians weighted higher than static signs), and raises an alert when
the configurable threshold is exceeded.
"""

from __future__ import annotations
from dataclasses import dataclass

CLASS_WEIGHT = {"pedestrian": 1.5, "cyclist": 1.3, "car": 1.0, "truck_bus": 1.1,
                "traffic_sign": 0.2, "traffic_light": 0.2}

ALERT_THRESHOLD = 0.55  # tuned empirically on validation clips (Section 7.6)


@dataclass
class RiskAssessment:
    track_id: int
    cls: str
    ttc: float
    risk: float
    alert: bool


def compute_ttc(distance_m: float, closing_speed_mps: float) -> float:
    if closing_speed_mps <= 0:
        return float("inf")
    return distance_m / closing_speed_mps


def risk_score(ttc: float, obj_class: str, lateral_offset_m: float) -> float:
    class_weight = CLASS_WEIGHT.get(obj_class, 1.0)
    proximity_factor = max(0.0, 1.0 - lateral_offset_m / 3.0)
    urgency = max(0.0, 1.0 - min(ttc, 5.0) / 5.0)
    return class_weight * proximity_factor * urgency


def assess_track(
    track_id: int,
    obj_class: str,
    distance_m: float,
    closing_speed_mps: float,
    lateral_offset_m: float,
    threshold: float = ALERT_THRESHOLD,
) -> RiskAssessment:
    ttc = compute_ttc(distance_m, closing_speed_mps)
    risk = risk_score(ttc, obj_class, lateral_offset_m)
    return RiskAssessment(track_id, obj_class, ttc, risk, alert=risk > threshold)


def px_velocity_to_metric(vx_px: float, vy_px: float, px_per_meter: float = 40.0):
    """
    Rough conversion from tracked pixel-space velocity to an approximate
    metric closing speed / lateral offset, used only for the demo pipeline
    where no real-world calibration (ground-plane homography) is available.
    """
    closing_speed_mps = max(0.0, -vy_px) / px_per_meter * 10.0  # objects moving 'up' the frame = closing
    lateral_offset_m = abs(vx_px) / px_per_meter
    return closing_speed_mps, lateral_offset_m


if __name__ == "__main__":
    r1 = assess_track(1, "pedestrian", distance_m=10.0, closing_speed_mps=5.0, lateral_offset_m=0.5)
    r2 = assess_track(2, "car", distance_m=40.0, closing_speed_mps=2.0, lateral_offset_m=3.0)
    for r in (r1, r2):
        print(r)
