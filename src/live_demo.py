import argparse
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from .pose_utils import FEATURE_DIM, PredictionSmoother, normalize_landmarks
from .form_feedback import ExerciseCounter, body_angles, feedback_for_label

try:
    from .models import PositionalEmbedding
except Exception:  # LSTM does not need this, but Transformer loading does.
    PositionalEmbedding = None


def result_to_raw_landmarks(result) -> np.ndarray:
    pts = []
    for lm in result.pose_landmarks.landmark:
        pts.append([lm.x, lm.y, lm.z])
    return np.asarray(pts, dtype=np.float32)


def landmarks_to_features(raw_landmarks_33x3: np.ndarray) -> np.ndarray:
    return normalize_landmarks(raw_landmarks_33x3.reshape(-1).astype(np.float32))


def put_text(frame, text, x, y, scale=0.7, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--coach", action="store_true", help="Show form feedback and rep counter")
    parser.add_argument("--show-angles", action="store_true", help="Show knee/elbow angles")
    args = parser.parse_args()

    custom_objects = {}
    if PositionalEmbedding is not None:
        custom_objects["PositionalEmbedding"] = PositionalEmbedding
    model = tf.keras.models.load_model(args.model_path, compile=False, custom_objects=custom_objects)

    with open(args.labels) as f:
        labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}

    buffer = deque(maxlen=args.seq_len)
    smoother = PredictionSmoother(window=args.smooth)
    counter = ExerciseCounter()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1 or check camera permissions.")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            label_text = "Collecting pose..."
            label = "Unknown"
            conf = 0.0
            raw_lms = None
            feedback = []
            reps, stage = counter.reps, counter.stage

            if result.pose_landmarks:
                mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                raw_lms = result_to_raw_landmarks(result)
                feats = landmarks_to_features(raw_lms)
                buffer.append(feats)
            else:
                buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))

            if len(buffer) == args.seq_len:
                seq = np.expand_dims(np.stack(buffer).astype(np.float32), axis=0)
                probs = model.predict(seq, verbose=0)[0]
                probs = smoother.update(probs)
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                if conf < args.confidence_threshold:
                    label = "Unknown"
                    label_text = f"Unknown: {conf:.2f}"
                else:
                    label = labels[pred_idx]
                    label_text = f"{label}: {conf:.2f}"

                if args.coach and raw_lms is not None and label != "Unknown":
                    reps, stage = counter.update(label, raw_lms)
                    feedback = feedback_for_label(label, raw_lms)

            # Header panel
            cv2.rectangle(frame, (10, 10), (760, 155 if args.coach else 70), (0, 0, 0), -1)
            put_text(frame, label_text, 20, 45, scale=1.0)

            if args.coach:
                put_text(frame, f"Reps: {reps} | Stage: {stage}", 20, 80, scale=0.75)
                if feedback:
                    put_text(frame, "Feedback: " + feedback[0], 20, 115, scale=0.65)
                    if len(feedback) > 1:
                        put_text(frame, feedback[1], 20, 145, scale=0.60)
                elif raw_lms is None:
                    put_text(frame, "Feedback: step back so full body is visible", 20, 115, scale=0.65)

            if args.show_angles and raw_lms is not None:
                angles = body_angles(raw_lms)
                h = frame.shape[0]
                cv2.rectangle(frame, (10, h - 80), (440, h - 10), (0, 0, 0), -1)
                put_text(frame, f"Knee: {angles['knee']:.0f} deg", 20, h - 50, scale=0.65)
                put_text(frame, f"Elbow: {angles['elbow']:.0f} deg", 20, h - 20, scale=0.65)

            cv2.imshow("AI Fitness Coach - Pose Action Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
