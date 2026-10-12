"""Pure feature functions for drowsiness detection.

This module is intentionally free of OpenCV / MediaPipe imports so every
function here is a pure, deterministic transform that can be unit tested
without a webcam. A "point" is any (x, y) pair (tuple, list or sequence of
two numbers). Distances are in whatever unit the points are in (we use
pixels at the call site, but the maths does not care).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Sequence, Tuple

Point = Sequence[float]  # (x, y)


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Straight-line distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye: Sequence[Point]) -> float:
    """Eye Aspect Ratio (EAR) from 6 eye landmarks.

    Points must be ordered ``[p1, p2, p3, p4, p5, p6]`` where ``p1`` and
    ``p4`` are the horizontal corners and (p2, p6) / (p3, p5) are the two
    vertical pairs:

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    A fully open eye is ~0.3; a closed eye drops toward ~0.1.
    """
    if len(eye) != 6:
        raise ValueError("eye_aspect_ratio expects exactly 6 points")

    p1, p2, p3, p4, p5, p6 = eye
    vertical = euclidean_distance(p2, p6) + euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def average_ear(left_eye: Sequence[Point], right_eye: Sequence[Point]) -> float:
    """Mean EAR of both eyes (each given as 6 ordered points)."""
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0


def mouth_aspect_ratio(mouth: Sequence[Point]) -> float:
    """Mouth Aspect Ratio (MAR) for yawn detection, from 8 landmarks.

    Points must be ordered::

        [left_corner, right_corner,
         upper1, lower1,   # vertical pair A
         upper2, lower2,   # vertical pair B (lip centre)
         upper3, lower3]   # vertical pair C

        MAR = (|u1-l1| + |u2-l2| + |u3-l3|) / (2 * |left_corner - right_corner|)

    A closed mouth is small (~0.1); a wide yawn pushes it above ~0.6.
    """
    if len(mouth) != 8:
        raise ValueError("mouth_aspect_ratio expects exactly 8 points")

    left_corner, right_corner, u1, l1, u2, l2, u3, l3 = mouth
    vertical = (
        euclidean_distance(u1, l1)
        + euclidean_distance(u2, l2)
        + euclidean_distance(u3, l3)
    )
    horizontal = euclidean_distance(left_corner, right_corner)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


class PerclosTracker:
    """Rolling PERCLOS = fraction of recent frames with eyes closed.

    PERCLOS (PERcentage of eyelid CLOSure) is a well established drowsiness
    measure: the proportion of time the eyes are closed over a sliding time
    window. We keep ``(timestamp, closed)`` samples in a deque and drop any
    sample older than ``window_seconds`` on every update.
    """

    def __init__(self, window_seconds: float = 30.0) -> None:
        self.window_seconds = window_seconds
        self._samples: Deque[Tuple[float, bool]] = deque()

    def update(self, timestamp: float, closed: bool) -> float:
        """Add a sample taken at ``timestamp`` and return current PERCLOS.

        ``closed`` is True when the eye was below the closed-EAR threshold on
        this frame. Returns a value in [0.0, 1.0].
        """
        self._samples.append((timestamp, bool(closed)))

        cutoff = timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        return self.value

    @property
    def value(self) -> float:
        """Current PERCLOS fraction over the window (0.0 if no samples)."""
        if not self._samples:
            return 0.0
        closed_count = sum(1 for _, closed in self._samples if closed)
        return closed_count / len(self._samples)


class ClosureTimer:
    """Tracks how long the eyes have been *continuously* closed.

    Fed one frame at a time; returns the duration (seconds) of the current
    unbroken run of closed frames. Resets to 0 as soon as a frame is open.
    """

    def __init__(self) -> None:
        self._closed_since: float | None = None

    def update(self, timestamp: float, closed: bool) -> float:
        """Return seconds the eyes have been closed up to ``timestamp``."""
        if closed:
            if self._closed_since is None:
                self._closed_since = timestamp
            return timestamp - self._closed_since
        self._closed_since = None
        return 0.0
