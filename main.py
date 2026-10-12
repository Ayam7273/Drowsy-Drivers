"""Real-time drowsy-driver detection from a webcam.

Pipeline:
    webcam frame  ->  MediaPipe Face Mesh landmarks
                  ->  EAR / MAR / PERCLOS features (src.detector)
                  ->  ALERT / TIRED / MICROSLEEP state
                  ->  on-screen overlay + per-frame CSV log

Run:
    python main.py                 # default webcam, logs to drowsiness_log.csv
    python main.py --camera 1      # pick another camera index
    python main.py --no-log        # disable CSV logging

Press 'q' or Esc to quit.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from typing import Optional

import cv2
import mediapipe as mp

from src.detector import DrowsinessDetector, State

# BGR colors for each state (OpenCV uses BGR, not RGB).
STATE_COLORS = {
    State.ALERT: (0, 200, 0),        # green
    State.TIRED: (0, 165, 255),      # orange
    State.MICROSLEEP: (0, 0, 255),   # red
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drowsy-driver detection (Phase 1)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument(
        "--log",
        default="drowsiness_log.csv",
        help="CSV log path (default drowsiness_log.csv)",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    parser.add_argument(
        "--calibration",
        type=float,
        default=10.0,
        help="Calibration window in seconds (default 10)",
    )
    return parser.parse_args()


def draw_overlay(frame, features, detector: DrowsinessDetector) -> None:
    """Draw the feature readouts and state banner onto ``frame`` in place."""
    height, width = frame.shape[:2]
    color = STATE_COLORS.get(features.state, (255, 255, 255))

    # Translucent banner at the top for the state text.
    banner_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if detector.calibrating:
        label = "CALIBRATING... keep eyes open & look ahead"
        banner_color = (200, 200, 200)
    else:
        label = features.state.value
        banner_color = color
    cv2.putText(
        frame, label, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_color, 2, cv2.LINE_AA
    )

    # Feature readouts, stacked below the banner.
    lines = [
        f"EAR:     {features.ear:.3f}",
        f"MAR:     {features.mar:.3f}",
        f"PERCLOS: {features.perclos * 100:5.1f}%",
    ]
    if features.yawning:
        lines.append("YAWN")
    if detector.closed_threshold is not None:
        lines.append(f"thr:     {detector.closed_threshold:.3f}")

    y = banner_h + 30
    for line in lines:
        cv2.putText(
            frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 30


def open_csv(path: Optional[str]):
    """Open the CSV log and write the header. Returns (file, writer) or (None, None)."""
    if path is None:
        return None, None
    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(["timestamp", "ear", "mar", "perclos", "state"])
    return handle, writer


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    detector = DrowsinessDetector(calibration_seconds=args.calibration)

    log_path = None if args.no_log else args.log
    csv_file, csv_writer = open_csv(log_path)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # better eye/iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start = time.monotonic()
    print("Starting drowsiness detection. Press 'q' or Esc to quit.")
    if log_path:
        print(f"Logging features to {log_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera; stopping.")
                break

            # Mirror the frame so it behaves like a mirror for the driver.
            frame = cv2.flip(frame, 1)
            timestamp = time.monotonic() - start

            # MediaPipe expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                height, width = frame.shape[:2]
                features = detector.process(landmarks, width, height, timestamp)
                draw_overlay(frame, features, detector)

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            f"{datetime.now().isoformat()}",
                            f"{features.ear:.4f}",
                            f"{features.mar:.4f}",
                            f"{features.perclos:.4f}",
                            features.state.value,
                        ]
                    )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Drowsy Driver Detection (Phase 1)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        if csv_file is not None:
            csv_file.close()
            print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
