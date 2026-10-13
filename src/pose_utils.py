from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


FEATURE_DIM = 33 * 3


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """Normalize pose landmarks to reduce camera-distance and body-position effects.

    Input shape can be (99,) or (T, 99). The layout is 33 landmarks with x, y, z.
    We center around the hip midpoint and scale by shoulder distance.
    """
    single = features.ndim == 1
    arr = features.reshape(1, -1) if single else features.copy()
    arr = arr.reshape(arr.shape[0], 33, 3)

    left_hip = arr[:, 23, :]
    right_hip = arr[:, 24, :]
    left_shoulder = arr[:, 11, :]
    right_shoulder = arr[:, 12, :]

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_dist = np.linalg.norm(left_shoulder[:, :2] - right_shoulder[:, :2], axis=1)
    shoulder_dist = np.maximum(shoulder_dist, 1e-6)

    arr = arr - hip_center[:, None, :]
    arr = arr / shoulder_dist[:, None, None]
    arr = arr.reshape(arr.shape[0], FEATURE_DIM)
    return arr[0] if single else arr


def sample_indices(total_frames: int, seq_len: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(seq_len, dtype=int)
    return np.linspace(0, total_frames - 1, seq_len).astype(int)


def extract_pose_sequence(video_path: str, seq_len: int = 48, min_visibility: float = 0.3) -> Optional[np.ndarray]:
    """Extract a fixed-length pose sequence from a video using MediaPipe Pose."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(sample_indices(total, seq_len).tolist()) if total > 0 else set()
    selected: List[np.ndarray] = []

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        frame_idx = 0
        last_valid: Optional[np.ndarray] = None
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            use_frame = frame_idx in idxs if total > 0 else frame_idx % max(1, int(cap.get(cv2.CAP_PROP_FPS)) // 8 or 1) == 0
            if use_frame:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    feats = []
                    visible_count = 0
                    for lm in result.pose_landmarks.landmark:
                        feats.extend([lm.x, lm.y, lm.z])
                        if lm.visibility >= min_visibility:
                            visible_count += 1
                    if visible_count >= 18:
                        last_valid = normalize_landmarks(np.array(feats, dtype=np.float32))
                if last_valid is not None:
                    selected.append(last_valid)
                else:
                    selected.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            frame_idx += 1
            if len(selected) >= seq_len:
                break

    cap.release()

    if len(selected) == 0:
        return None
    while len(selected) < seq_len:
        selected.append(selected[-1])
    return np.stack(selected[:seq_len]).astype(np.float32)


class PredictionSmoother:
    def __init__(self, window: int = 5):
        self.window = max(1, window)
        self.values = deque(maxlen=self.window)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.values.append(probs)
        return np.mean(np.stack(list(self.values)), axis=0)


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees for three 2D/3D points."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))
