"""
calibration.py
Camera Calibration & Pre-processing Module
(Report Section 5.1 / 7.2)

Uses a planar chessboard pattern (Zhang's method) to estimate the camera's
intrinsic matrix K and distortion coefficients D, then provides an
undistort() helper that is applied to every incoming frame before any
further processing in the pipeline.
"""

from __future__ import annotations
import glob
import os
import numpy as np
import cv2


class CameraCalibrator:
    def __init__(self, pattern_size: tuple[int, int] = (9, 6), square_size: float = 1.0):
        """
        pattern_size: number of inner corners per chessboard row/column (cols, rows)
        square_size:  physical size of one chessboard square (any consistent unit,
                      only matters if you need metric extrinsics)
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.K: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.new_K: np.ndarray | None = None
        self.reprojection_error: float | None = None

    def calibrate(self, chessboard_images_glob: str) -> tuple[np.ndarray, np.ndarray]:
        """Run full calibration over a folder of chessboard images."""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        objpoints, imgpoints = [], []
        image_size = None
        files = sorted(glob.glob(chessboard_images_glob))
        if not files:
            raise FileNotFoundError(f"No calibration images matched: {chessboard_images_glob}")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for fname in files:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) < 3:
            raise RuntimeError(
                f"Only {len(objpoints)} usable chessboard images found; need at least 3."
            )

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        self.K, self.D = K, D
        self.reprojection_error = self._compute_reprojection_error(
            objpoints, imgpoints, rvecs, tvecs, K, D
        )
        return K, D

    @staticmethod
    def _compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, D) -> float:
        total_error, total_points = 0.0, 0
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            error = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
            total_error += error
            total_points += 1
        return total_error / max(total_points, 1)

    def save(self, path: str = "calib_params.npz") -> None:
        if self.K is None:
            raise RuntimeError("Calibrate before saving.")
        np.savez(path, K=self.K, D=self.D)

    def load(self, path: str = "calib_params.npz") -> None:
        data = np.load(path)
        self.K, self.D = data["K"], data["D"]

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Undistort a frame using the calibrated intrinsics (Section 5.1)."""
        if self.K is None or self.D is None:
            raise RuntimeError("Camera not calibrated. Call calibrate() or load() first.")
        h, w = frame.shape[:2]
        if self.new_K is None:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
        return cv2.undistort(frame, self.K, self.D, None, self.new_K)


def synthetic_calibration(image_size=(640, 480)) -> CameraCalibrator:
    """
    Fallback used by the demo pipeline when no physical chessboard images are
    available: returns a plausible, mildly-distorted intrinsic matrix so the
    rest of the pipeline (undistort -> detect -> track) can run end-to-end.
    """
    w, h = image_size
    fx = fy = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([-0.05, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    cal = CameraCalibrator()
    cal.K, cal.D = K, D
    cal.reprojection_error = 0.0
    return cal


if __name__ == "__main__":
    # Example / smoke test using a synthetic calibration (no chessboard images required)
    cal = synthetic_calibration()
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    undistorted = cal.undistort(dummy_frame)
    print("K =\n", cal.K)
    print("D =", cal.D)
    print("Undistorted frame shape:", undistorted.shape)
