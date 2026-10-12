"""Landmark extraction + drowsiness state classification.

This sits one layer above :mod:`src.features`. It knows about MediaPipe Face
Mesh landmark *indices* and turns a raw landmark list into the EAR/MAR/PERCLOS
features, then classifies the driver's state. It still does not import OpenCV
or MediaPipe: callers pass in the normalized landmark list (each landmark has
``.x`` / ``.y`` in 0..1) plus the frame size, so this module stays easy to
test with simple stand-in objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .features import (
    ClosureTimer,
    PerclosTracker,
    average_ear,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)

# --- MediaPipe Face Mesh landmark indices --------------------------------
# Each eye is 6 points ordered [p1, p2, p3, p4, p5, p6] for the EAR formula:
# p1/p4 = horizontal corners, (p2,p6) and (p3,p5) = vertical pairs.
RIGHT_EYE_IDX: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# Mouth: [left_corner, right_corner, u1, l1, u2, l2, u3, l3] for the MAR formula.
MOUTH_IDX: Tuple[int, ...] = (78, 308, 82, 87, 13, 14, 312, 317)

Point = Tuple[float, float]


class State(str, Enum):
    """Driver state. Inherits from str so it logs/prints as plain text."""

    ALERT = "ALERT"
    TIRED = "TIRED"
    MICROSLEEP = "MICROSLEEP"


@dataclass
class Thresholds:
    """Tunable classification thresholds (sensible defaults)."""

    # Fraction of the calibrated open-eye EAR below which an eye is "closed".
    closed_ratio: float = 0.75
    # Continuous eye closure beyond this many seconds => MICROSLEEP.
    microsleep_seconds: float = 2.0
    # PERCLOS above this fraction => TIRED.
    perclos_tired: float = 0.15
    # MAR above this => mouth wide open (yawn) => contributes to TIRED.
    yawn_mar: float = 0.6


@dataclass
class Features:
    """Per-frame feature bundle returned by the detector."""

    ear: float
    mar: float
    perclos: float
    closed_duration: float
    yawning: bool
    state: State


@dataclass
class _Calibration:
    """Collects open-eye EAR during the warm-up period."""

    duration: float = 10.0
    closed_ratio: float = 0.75
    _samples: List[float] = field(default_factory=list)
    _start: Optional[float] = None
    baseline_ear: Optional[float] = None
    closed_threshold: Optional[float] = None

    def is_active(self, timestamp: float) -> bool:
        """True while still within the calibration window."""
        if self._start is None:
            self._start = timestamp
        return (timestamp - self._start) < self.duration

    def add(self, ear: float) -> None:
        """Record one open-eye EAR sample (assumes the user is alert)."""
        if ear > 0:
            self._samples.append(ear)

    def finalize(self) -> None:
        """Compute baseline EAR and the closed threshold from samples."""
        if self._samples:
            self.baseline_ear = sum(self._samples) / len(self._samples)
            self.closed_threshold = self.baseline_ear * self.closed_ratio


def _to_pixel(landmark, width: int, height: int) -> Point:
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return (landmark.x * width, landmark.y * height)


def extract_points(
    landmarks: Sequence,
    width: int,
    height: int,
    indices: Sequence[int],
) -> List[Point]:
    """Pick ``indices`` from ``landmarks`` as pixel-space points."""
    return [_to_pixel(landmarks[i], width, height) for i in indices]


class DrowsinessDetector:
    """Stateful per-session detector: calibrate, then classify each frame.

    Typical use (one instance per run)::

        det = DrowsinessDetector()
        features = det.process(landmarks, width, height, timestamp)

    ``landmarks`` is the MediaPipe Face Mesh landmark list (length 468),
    each item exposing ``.x`` and ``.y`` in the 0..1 range.
    """

    def __init__(
        self,
        thresholds: Optional[Thresholds] = None,
        calibration_seconds: float = 10.0,
        perclos_window_seconds: float = 30.0,
    ) -> None:
        self.thresholds = thresholds or Thresholds()
        self._calibration = _Calibration(
            duration=calibration_seconds,
            closed_ratio=self.thresholds.closed_ratio,
        )
        self._perclos = PerclosTracker(window_seconds=perclos_window_seconds)
        self._closure = ClosureTimer()

    @property
    def calibrating(self) -> bool:
        """True until the calibration window has elapsed and finalized."""
        return self._calibration.closed_threshold is None

    @property
    def closed_threshold(self) -> Optional[float]:
        """The calibrated closed-eye EAR threshold (None until calibrated)."""
        return self._calibration.closed_threshold

    @property
    def baseline_ear(self) -> Optional[float]:
        """The calibrated open-eye EAR baseline (None until calibrated)."""
        return self._calibration.baseline_ear

    def compute_raw_features(
        self, landmarks: Sequence, width: int, height: int
    ) -> Tuple[float, float]:
        """Return (ear, mar) for one frame's landmarks. Pure given inputs."""
        left_eye = extract_points(landmarks, width, height, LEFT_EYE_IDX)
        right_eye = extract_points(landmarks, width, height, RIGHT_EYE_IDX)
        mouth = extract_points(landmarks, width, height, MOUTH_IDX)

        ear = average_ear(left_eye, right_eye)
        mar = mouth_aspect_ratio(mouth)
        return ear, mar

    def classify(
        self, ear: float, mar: float, perclos: float, closed_duration: float
    ) -> State:
        """Map features to a state using the simple threshold rules."""
        t = self.thresholds
        if closed_duration > t.microsleep_seconds:
            return State.MICROSLEEP
        yawning = mar > t.yawn_mar
        if perclos > t.perclos_tired or yawning:
            return State.TIRED
        return State.ALERT

    def process(
        self, landmarks: Sequence, width: int, height: int, timestamp: float
    ) -> Features:
        """Full per-frame step: features + calibration + classification."""
        ear, mar = self.compute_raw_features(landmarks, width, height)

        # During warm-up we assume the driver is alert with eyes open and
        # gather a personal EAR baseline. We report ALERT and PERCLOS 0.
        if self._calibration.is_active(timestamp):
            self._calibration.add(ear)
            return Features(
                ear=ear,
                mar=mar,
                perclos=0.0,
                closed_duration=0.0,
                yawning=mar > self.thresholds.yawn_mar,
                state=State.ALERT,
            )

        # First frame after the window closes: lock in the thresholds.
        if self._calibration.closed_threshold is None:
            self._calibration.finalize()

        threshold = self._calibration.closed_threshold
        # Fallback if calibration gathered nothing (e.g. no face seen yet).
        if threshold is None:
            threshold = 0.2

        closed = ear < threshold
        perclos = self._perclos.update(timestamp, closed)
        closed_duration = self._closure.update(timestamp, closed)
        state = self.classify(ear, mar, perclos, closed_duration)

        return Features(
            ear=ear,
            mar=mar,
            perclos=perclos,
            closed_duration=closed_duration,
            yawning=mar > self.thresholds.yawn_mar,
            state=state,
        )
