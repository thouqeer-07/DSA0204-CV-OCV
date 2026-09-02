"""
make_demo_video.py
Generates a short synthetic driving-scene video (moving 'car' and
'pedestrian' rectangles, one temporarily occluded by a 'truck') so the
full pipeline can be demonstrated end-to-end without external datasets
(KITTI/BDD100K are referenced in the report as the intended real data
sources -- Section 3.3 / 7.1).
"""
import cv2
import numpy as np


def generate_demo_video(path: str = "sample_data/demo.mp4", n_frames: int = 90,
                         size=(640, 480), fps: int = 30) -> str:
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(n_frames):
        frame = np.full((h, w, 3), (200, 200, 200), dtype=np.uint8)
        cv2.rectangle(frame, (0, int(h * 0.6)), (w, h), (60, 60, 60), -1)  # road
        cv2.line(frame, (int(w * 0.25), h), (int(w * 0.45), int(h * 0.6)), (255, 255, 255), 3)
        cv2.line(frame, (int(w * 0.75), h), (int(w * 0.55), int(h * 0.6)), (255, 255, 255), 3)

        # Car moving left -> right
        car_x = int(50 + i * 4)
        cv2.rectangle(frame, (car_x, 320), (car_x + 80, 380), (200, 60, 30), -1)

        # Truck (static occluder) in the middle of the frame
        cv2.rectangle(frame, (300, 280), (420, 400), (90, 90, 90), -1)

        # Pedestrian walking behind the truck for frames 30-50 (occluded), else visible
        ped_x = int(250 + i * 2)
        ped_visible = not (30 <= i <= 50)
        if ped_visible:
            cv2.rectangle(frame, (ped_x, 300), (ped_x + 25, 380), (30, 30, 220), -1)

        writer.write(frame)

    writer.release()
    return path


if __name__ == "__main__":
    out = generate_demo_video()
    print(f"Demo video written to {out}")
