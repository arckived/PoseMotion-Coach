"""Real-time exercise form feedback and rep counting helpers.

The classifier predicts the action from temporal pose keypoints.
This file adds simple rule-based coaching on top of the prediction using
MediaPipe's 33 body landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .pose_utils import angle_3pt


# MediaPipe Pose landmark indexes used here:
# shoulders: 11, 12 | elbows: 13, 14 | wrists: 15, 16
# hips: 23, 24 | knees: 25, 26 | ankles: 27, 28


def _avg_angle(lms: np.ndarray, triples: List[Tuple[int, int, int]]) -> float:
    vals = []
    for a, b, c in triples:
        vals.append(angle_3pt(lms[a, :2], lms[b, :2], lms[c, :2]))
    return float(np.mean(vals)) if vals else 0.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def body_angles(landmarks_33x3: np.ndarray) -> Dict[str, float]:
    """Return useful form angles from raw MediaPipe landmarks."""
    lms = landmarks_33x3
    return {
        "knee": _avg_angle(lms, [(23, 25, 27), (24, 26, 28)]),
        "left_knee": angle_3pt(lms[23, :2], lms[25, :2], lms[27, :2]),
        "right_knee": angle_3pt(lms[24, :2], lms[26, :2], lms[28, :2]),
        "elbow": _avg_angle(lms, [(11, 13, 15), (12, 14, 16)]),
        "left_elbow": angle_3pt(lms[11, :2], lms[13, :2], lms[15, :2]),
        "right_elbow": angle_3pt(lms[12, :2], lms[14, :2], lms[16, :2]),
        "hip": _avg_angle(lms, [(11, 23, 25), (12, 24, 26)]),
    }


def squat_feedback(landmarks_33x3: np.ndarray) -> List[str]:
    feedback: List[str] = []
    angles = body_angles(landmarks_33x3)
    knee = angles["knee"]

    if knee > 130:
        feedback.append("Go deeper for a full squat")
    elif knee < 65:
        feedback.append("Very deep squat; keep control")
    else:
        feedback.append("Good squat depth")

    shoulder_mid = (landmarks_33x3[11] + landmarks_33x3[12]) / 2
    hip_mid = (landmarks_33x3[23] + landmarks_33x3[24]) / 2
    torso_tilt = abs(shoulder_mid[0] - hip_mid[0])
    if torso_tilt > 0.18:
        feedback.append("Keep chest more upright")

    return feedback


def pushup_feedback(landmarks_33x3: np.ndarray) -> List[str]:
    feedback: List[str] = []
    angles = body_angles(landmarks_33x3)
    elbow = angles["elbow"]
    hip = angles["hip"]

    if elbow > 145:
        feedback.append("Lower your chest more")
    elif elbow < 60:
        feedback.append("Very low; push back up with control")
    else:
        feedback.append("Good push-up depth")

    if hip < 150:
        feedback.append("Keep hips and torso straighter")
    return feedback


def lunge_feedback(landmarks_33x3: np.ndarray) -> List[str]:
    angles = body_angles(landmarks_33x3)
    front_knee = min(angles["left_knee"], angles["right_knee"])
    if front_knee > 125:
        return ["Go lower into the lunge"]
    if front_knee < 65:
        return ["Too deep; keep balance and control"]
    return ["Good lunge depth"]


def jumping_jack_feedback(landmarks_33x3: np.ndarray) -> List[str]:
    lms = landmarks_33x3
    shoulder_width = max(_dist(lms[11], lms[12]), 1e-6)
    ankle_width = _dist(lms[27], lms[28]) / shoulder_width
    wrists_above_shoulders = lms[15, 1] < lms[11, 1] and lms[16, 1] < lms[12, 1]

    feedback: List[str] = []
    if ankle_width < 1.4:
        feedback.append("Step feet wider")
    if not wrists_above_shoulders:
        feedback.append("Raise arms higher")
    if not feedback:
        feedback.append("Good jumping jack extension")
    return feedback


def pullup_feedback(landmarks_33x3: np.ndarray) -> List[str]:
    angles = body_angles(landmarks_33x3)
    elbow = angles["elbow"]
    if elbow > 145:
        return ["Arms extended; pull upward"]
    if elbow < 80:
        return ["Good pull-up contraction"]
    return ["Keep pulling with control"]


def feedback_for_label(label: str, landmarks_33x3: np.ndarray) -> List[str]:
    name = label.lower()
    if "squat" in name:
        return squat_feedback(landmarks_33x3)
    if "push" in name:
        return pushup_feedback(landmarks_33x3)
    if "lunge" in name:
        return lunge_feedback(landmarks_33x3)
    if "jumpingjack" in name or "jumping_jack" in name or "jumping jack" in name:
        return jumping_jack_feedback(landmarks_33x3)
    if "pull" in name:
        return pullup_feedback(landmarks_33x3)
    return []


@dataclass
class ExerciseCounter:
    """Simple landmark-angle rep counter for live webcam demo."""

    reps: int = 0
    stage: str = "unknown"
    last_label: str = ""

    def reset_if_label_changed(self, label: str) -> None:
        if label != self.last_label:
            self.reps = 0
            self.stage = "unknown"
            self.last_label = label

    def update(self, label: str, landmarks_33x3: np.ndarray) -> Tuple[int, str]:
        self.reset_if_label_changed(label)
        name = label.lower()
        angles = body_angles(landmarks_33x3)

        if "squat" in name:
            return self._count_threshold(angles["knee"], down_below=105, up_above=155)

        if "push" in name:
            return self._count_threshold(angles["elbow"], down_below=100, up_above=155)

        if "lunge" in name:
            front_knee = min(angles["left_knee"], angles["right_knee"])
            return self._count_threshold(front_knee, down_below=105, up_above=150)

        if "pull" in name:
            # Pull-up is reversed: down = arms extended, up = elbows bent.
            elbow = angles["elbow"]
            if elbow > 150:
                self.stage = "down"
            elif elbow < 85 and self.stage == "down":
                self.stage = "up"
                self.reps += 1
            return self.reps, self.stage

        if "jumpingjack" in name or "jumping_jack" in name or "jumping jack" in name:
            return self._count_jumping_jack(landmarks_33x3)

        return self.reps, self.stage

    def _count_threshold(self, angle: float, down_below: float, up_above: float) -> Tuple[int, str]:
        if angle < down_below:
            self.stage = "down"
        elif angle > up_above and self.stage == "down":
            self.stage = "up"
            self.reps += 1
        return self.reps, self.stage

    def _count_jumping_jack(self, landmarks_33x3: np.ndarray) -> Tuple[int, str]:
        lms = landmarks_33x3
        shoulder_width = max(_dist(lms[11], lms[12]), 1e-6)
        ankle_width = _dist(lms[27], lms[28]) / shoulder_width
        wrists_above_shoulders = lms[15, 1] < lms[11, 1] and lms[16, 1] < lms[12, 1]
        is_open = ankle_width > 1.6 and wrists_above_shoulders

        if is_open:
            self.stage = "open"
        elif self.stage == "open":
            self.stage = "closed"
            self.reps += 1
        return self.reps, self.stage
