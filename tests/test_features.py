"""Unit tests for the pure feature functions.

Run with:  python -m pytest   (or)   python tests/test_features.py
These need no webcam, OpenCV or MediaPipe -- that's the point of keeping
src/features.py pure.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (  # noqa: E402
    ClosureTimer,
    PerclosTracker,
    average_ear,
    euclidean_distance,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_euclidean_distance():
    assert _approx(euclidean_distance((0, 0), (3, 4)), 5.0)
    assert _approx(euclidean_distance((1, 1), (1, 1)), 0.0)


def test_eye_aspect_ratio_open_vs_closed():
    # Horizontal span 10; vertical pairs span 3 each when "open".
    open_eye = [(0, 0), (3, 3), (7, 3), (10, 0), (7, -3), (3, -3)]
    # Same eye but vertical pairs almost touching => "closed".
    closed_eye = [(0, 0), (3, 0.2), (7, 0.2), (10, 0), (7, -0.2), (3, -0.2)]

    ear_open = eye_aspect_ratio(open_eye)
    ear_closed = eye_aspect_ratio(closed_eye)

    assert ear_open > ear_closed
    assert ear_closed < 0.1


def test_eye_aspect_ratio_requires_six_points():
    try:
        eye_aspect_ratio([(0, 0)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong point count")


def test_average_ear():
    eye = [(0, 0), (3, 3), (7, 3), (10, 0), (7, -3), (3, -3)]
    single = eye_aspect_ratio(eye)
    assert _approx(average_ear(eye, eye), single)


def test_mouth_aspect_ratio_closed_vs_yawn():
    # order: [left_corner, right_corner, u1,l1, u2,l2, u3,l3], horizontal span 20.
    closed = [(0, 0), (20, 0), (6, 1), (6, -1), (10, 1), (10, -1), (14, 1), (14, -1)]
    yawn = [(0, 0), (20, 0), (6, 8), (6, -8), (10, 9), (10, -9), (14, 8), (14, -8)]

    assert mouth_aspect_ratio(yawn) > mouth_aspect_ratio(closed)
    assert mouth_aspect_ratio(closed) < 0.2


def test_perclos_rolling_window():
    tracker = PerclosTracker(window_seconds=10.0)
    # 10 open frames over t=0..9 -> PERCLOS 0.
    for t in range(10):
        tracker.update(float(t), closed=False)
    assert _approx(tracker.value, 0.0)

    # Now 10 closed frames t=10..19. By t=19 the window (>=9.0) holds a mix,
    # and old open frames drop out.
    for t in range(10, 20):
        tracker.update(float(t), closed=True)
    assert tracker.value > 0.0

    # Far in the future, all-closed only.
    for t in range(100, 110):
        tracker.update(float(t), closed=True)
    assert _approx(tracker.value, 1.0)


def test_closure_timer_resets_on_open():
    timer = ClosureTimer()
    assert _approx(timer.update(0.0, closed=True), 0.0)
    assert _approx(timer.update(1.5, closed=True), 1.5)
    assert _approx(timer.update(3.0, closed=True), 3.0)
    # An open frame resets the run.
    assert _approx(timer.update(3.5, closed=False), 0.0)
    assert _approx(timer.update(4.0, closed=True), 0.0)


if __name__ == "__main__":
    # Minimal runner so the file works without pytest installed.
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
