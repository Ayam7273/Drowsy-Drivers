"""Live demo: drowsiness detection + graded alerts + SIMULATED autopilot handover.

    python demo.py                 # webcam 0, uses models/*.joblib if present
    python demo.py --rules-only    # skip the ML models (pure EAR/MAR/PERCLOS rules)
    python demo.py --video clip.mp4 --mute

Keys (click the video window first):
    SPACE  acknowledge the "are you awake?" prompt
    R      hand control back to the driver (only works if the driver looks ALERT)
    T      toggle SIMULATED tiredness   (to demo the alerts without dozing off)
    SPACE and R also switch any simulation off
    M      toggle SIMULATED microsleep
    K      recalibrate (keep eyes open, look ahead)
    Q/Esc  quit

How the pieces fit:
    EAR / PERCLOS rules (src.detector) decide whether the eyes are closed.
    A Random Forest on mouth geometry (models/yawn_rf.joblib) flags yawns.
    A Random Forest on HOG eye crops (models/eye_rf.joblib) gives a second
    opinion on eye state; it is shown and logged but does not drive the state,
    because it was trained on dataset crops and live crops may differ.
    src.escalation turns the state into WARNING -> PROMPT -> TAKEOVER.
    Nothing here controls a real vehicle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.detector import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    DrowsinessDetector,
    State,
    extract_points,
)
from src.escalation import Escalation, EscalationConfig, Level, VehicleSim
from src.features import PerclosTracker, euclidean_distance

try:  # sound is Windows-only; fall back to silence elsewhere
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

LEVEL_COLORS = {  # BGR
    Level.NONE: (0, 170, 0),
    Level.WARNING: (0, 165, 255),
    Level.PROMPT: (0, 0, 255),
    Level.TAKEOVER: (255, 170, 40),
}


# ---------------------------------------------------------------- ML helpers
class MLModels:
    """Optional trained models. Missing files simply disable that model."""

    def __init__(self, models_dir: Path, enabled: bool = True) -> None:
        self.yawn = self.eye = None
        if not enabled:
            return
        try:
            import joblib
        except ImportError:
            print("joblib not installed - running rules only.")
            return
        for name in ("yawn", "eye"):
            path = models_dir / f"{name}_rf.joblib"
            if path.exists():
                try:
                    setattr(self, name, joblib.load(path))
                    print(f"Loaded {path}")
                except Exception as exc:  # corrupt file, version mismatch...
                    print(f"Could not load {path}: {exc}")
            else:
                print(f"{path} not found - {name} ML disabled.")
        if self.eye is not None:
            self._hog = cv2.HOGDescriptor(*self.eye["hog_params"])

    def yawn_probability(self, landmarks, width: int, height: int,
                         detector: DrowsinessDetector) -> Optional[float]:
        if self.yawn is None:
            return None
        from prepare_yawn_features import landmark_features
        try:
            feats = landmark_features(landmarks, width, height, detector)
        except ValueError:
            return None
        row = np.array([[feats[name] for name in self.yawn["features"]]], dtype=float)
        return float(self.yawn["model"].predict_proba(row)[0, 1])

    def eye_closed_probability(self, gray, landmarks, width: int, height: int) -> Optional[float]:
        if self.eye is None:
            return None
        size = self.eye["img_size"]
        feats = []
        for idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            crop = eye_crop(gray, landmarks, idx, width, height, size)
            if crop is not None:
                feats.append(self._hog.compute(cv2.equalizeHist(crop)).ravel())
        if not feats:
            return None
        proba = self.eye["model"].predict_proba(np.array(feats, dtype=np.float32))[:, 1]
        return float(proba.mean())


def eye_crop(gray, landmarks, idx, width: int, height: int, size: int):
    """Square grayscale crop around one eye, resized to ``size`` x ``size``."""
    pts = extract_points(landmarks, width, height, idx)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    side = 1.9 * euclidean_distance(pts[0], pts[3])
    half = int(side / 2)
    if half < 4:
        return None
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = int(cx) + half, int(cy) + half
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def fuse_state(rule_state: State, ml_yawn: Optional[float], threshold: float = 0.5) -> State:
    """Hybrid decision: an ML yawn raises ALERT to TIRED; it never lowers a state."""
    if ml_yawn is not None and ml_yawn > threshold and rule_state == State.ALERT:
        return State.TIRED
    return rule_state


# -------------------------------------------------------------------- sound
class Alarm:
    """Level-dependent beeps on a background thread (never blocks the video)."""

    def __init__(self, muted: bool) -> None:
        self.muted = muted or winsound is None
        self._last = -99.0
        self._prev = Level.NONE

    def tick(self, level: Level, now: float) -> None:
        if self.muted:
            return
        if level == Level.TAKEOVER and self._prev != Level.TAKEOVER:
            self._beep(500, 700)
        elif level == Level.WARNING and now - self._last >= 2.0:
            self._beep(800, 150)
            self._last = now
        elif level == Level.PROMPT and now - self._last >= 0.6:
            self._beep(1600, 250)
            self._last = now
        self._prev = level

    @staticmethod
    def _beep(freq: int, ms: int) -> None:
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()


# ----------------------------------------------------------------- the demo
class Demo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.models = MLModels(Path(args.models), enabled=not args.rules_only)
        self.calibration = args.calibration
        self.detector = DrowsinessDetector(calibration_seconds=args.calibration)
        self.esc = Escalation(EscalationConfig(prompt_timeout_seconds=args.prompt_timeout))
        self.vehicle = VehicleSim()
        self.alarm = Alarm(args.mute)
        self.sim_tired = False
        self.sim_micro = False
        self.ml_cache: Tuple[Optional[float], Optional[float]] = (None, None)
        self.frame_no = 0
        self.last_now: Optional[float] = None
        self.time_in_level: Dict[Level, float] = {lv: 0.0 for lv in Level}
        self.proc_ms: List[float] = []
        self.rows: List[list] = []

    def recalibrate(self) -> None:
        self.detector = DrowsinessDetector(calibration_seconds=self.calibration)

    # one frame in, annotated frame out
    def process(self, frame, landmarks, now: float, key: int) -> np.ndarray:
        t0 = time.perf_counter()
        height, width = frame.shape[:2]
        dt = 0.0 if self.last_now is None else max(0.0, now - self.last_now)
        self.last_now = now

        ack = key == 32
        if key in (ord("t"), ord("T")):
            self.sim_tired = not self.sim_tired
            self.sim_micro = False  # one simulation at a time
        if key in (ord("m"), ord("M")):
            self.sim_micro = not self.sim_micro
            self.sim_tired = False
        if ack or key in (ord("r"), ord("R")):
            # Answering the prompt or taking control back means "I am awake":
            # switch the simulation off so the real camera state is used again.
            self.sim_tired = self.sim_micro = False
        if key in (ord("k"), ord("K")):
            self.recalibrate()

        feats = None
        rule_state: Optional[State] = None
        if landmarks is not None:
            feats = self.detector.process(landmarks, width, height, now)
            rule_state = feats.state
            if self.frame_no % self.args.ml_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.ml_cache = (
                    self.models.yawn_probability(landmarks, width, height, self.detector),
                    self.models.eye_closed_probability(gray, landmarks, width, height),
                )
        yawn_p, eye_p = self.ml_cache if landmarks is not None else (None, None)

        state = None if rule_state is None else fuse_state(rule_state, yawn_p)
        if self.detector.calibrating and landmarks is not None:
            state = State.ALERT  # never escalate while calibrating
        if self.sim_micro:
            state = State.MICROSLEEP
        elif self.sim_tired:
            state = State.TIRED

        if key in (ord("r"), ord("R")):
            # Resuming needs eyes open *right now*; PERCLOS would stay high for
            # up to 30 s after a long closure and block the hand-back.
            if self.sim_micro or self.sim_tired or feats is None:
                instant = state
            else:
                threshold = self.detector.closed_threshold or 0.0
                instant = State.ALERT if feats.ear >= threshold and not feats.yawning else State.TIRED
            if self.esc.resume(instant, now):
                # Fresh start for the rolling PERCLOS window after a hand-back.
                self.detector._perclos = PerclosTracker(window_seconds=30.0)
        level = self.esc.update(state, now, acknowledged=ack)
        speed = self.vehicle.step(dt, level)
        self.alarm.tick(level, now)
        self.time_in_level[level] += dt

        self._draw(frame, feats, state, level, speed, yawn_p, eye_p, now)
        self.proc_ms.append((time.perf_counter() - t0) * 1000)
        self.rows.append([
            f"{now:.2f}",
            "" if feats is None else f"{feats.ear:.4f}",
            "" if feats is None else f"{feats.mar:.4f}",
            "" if feats is None else f"{feats.perclos:.4f}",
            "" if rule_state is None else rule_state.value,
            "" if eye_p is None else f"{eye_p:.3f}",
            "" if yawn_p is None else f"{yawn_p:.3f}",
            "NO_FACE" if state is None else state.value,
            level.name, f"{speed:.1f}",
        ])
        self.frame_no += 1
        return frame

    def _draw(self, frame, feats, state, level, speed, yawn_p, eye_p, now) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = LEVEL_COLORS[level]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if level == Level.TAKEOVER:
            title = "AUTOPILOT ENGAGED (SIMULATED)"
        elif level == Level.PROMPT:
            title = f"ARE YOU AWAKE? PRESS SPACE ({self.esc.prompt_remaining(now):.1f}s)"
        elif level == Level.WARNING:
            title = "WARNING: DRIVER DROWSY"
        elif self.detector.calibrating:
            title = "CALIBRATING - eyes open, look ahead"
            color = (200, 200, 200)
        else:
            title = "DRIVER ALERT"
        cv2.putText(frame, title, (12, 46), font, 0.95, color, 2, cv2.LINE_AA)

        if level in (Level.PROMPT, Level.TAKEOVER):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 10)
        if level == Level.TAKEOVER:
            msg = "STOPPED SAFELY" if self.vehicle.stopped else "slowing down, hazards on"
            cv2.putText(frame, msg, (12, 105), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press R when alert to take back control", (12, h - 20),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        lines = [f"Speed: {speed:5.1f} km/h",
                 f"State: {'NO FACE' if state is None else state.value}"]
        if feats is not None:
            lines += [f"EAR {feats.ear:.3f}  MAR {feats.mar:.3f}",
                      f"PERCLOS {feats.perclos * 100:4.1f}%"]
        if eye_p is not None:
            lines.append(f"ML eye closed: {eye_p:.2f}")
        if yawn_p is not None:
            lines.append(f"ML yawn: {yawn_p:.2f}")
        if self.sim_tired or self.sim_micro:
            lines.append("SIMULATED: " + ("MICROSLEEP (M=off)" if self.sim_micro else "TIRED (T=off)"))
        y = h - 20 - 26 * len(lines) if level != Level.TAKEOVER else 140
        for line in lines:
            cv2.putText(frame, line, (12, y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 26

    def summary(self, wall_seconds: float) -> dict:
        ms = np.array(self.proc_ms) if self.proc_ms else np.array([0.0])
        return {
            "frames": self.frame_no,
            "wall_seconds": round(wall_seconds, 1),
            "average_fps": round(self.frame_no / wall_seconds, 1) if wall_seconds else 0,
            "processing_ms_mean": round(float(ms.mean()), 1),
            "processing_ms_p95": round(float(np.percentile(ms, 95)), 1),
            "seconds_in_level": {lv.name: round(s, 1) for lv, s in self.time_in_level.items()},
            "events": [{"time": round(e.time, 1), "from": e.old.name, "to": e.new.name,
                        "reason": e.reason} for e in self.esc.events],
            "ml_models": {"yawn": self.models.yawn is not None, "eye": self.models.eye is not None},
        }

    def write_log(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "ear", "mar", "perclos", "rule_state", "ml_eye_closed_p",
                        "ml_yawn_p", "final_state", "level", "speed_kmh"])
            w.writerows(self.rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drowsy driver demo with simulated autopilot")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="use a video file instead of the webcam")
    p.add_argument("--models", default="models", help="folder with the .joblib models")
    p.add_argument("--rules-only", action="store_true", help="do not load ML models")
    p.add_argument("--calibration", type=float, default=10.0)
    p.add_argument("--prompt-timeout", type=float, default=5.0,
                   help="seconds to answer the prompt before autopilot engages")
    p.add_argument("--ml-every", type=int, default=2, help="run the ML models every N frames")
    p.add_argument("--mute", action="store_true", help="no beeps")
    p.add_argument("--log", default="demo_log.csv")
    p.add_argument("--summary", default="results/demo_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source. Close other apps using the "
                         "camera or try --camera 1.")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    demo = Demo(args)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("Demo running. SPACE=ack  R=resume  T=sim tired  M=sim microsleep  K=recalibrate  Q=quit")
    start = time.monotonic()
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.video:
                frame = cv2.flip(frame, 1)
            now = (n / video_fps) if args.video else (time.monotonic() - start)
            n += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)
            landmarks = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            demo.process(frame, landmarks, now, key)
            if landmarks is None and not (demo.sim_tired or demo.sim_micro):
                cv2.putText(frame, "No face detected", (12, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Drowsy Driver Demo (simulated autopilot)", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        wall = time.monotonic() - start
        demo.write_log(args.log)
        summ = demo.summary(wall)
        try:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(json.dumps(summ, indent=2), encoding="utf-8")
        except OSError:
            pass
        print(json.dumps(summ, indent=2))
        print(f"Log saved to {args.log}; summary saved to {args.summary}")


if __name__ == "__main__":
    main()
