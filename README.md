# Drowsy-Driver Detection — Phase 1

A real-time webcam pipeline that estimates driver drowsiness from facial
landmarks. Phase 1 is classic computer vision only — **no deep learning yet**.
It computes eye/mouth ratios, tracks PERCLOS over a rolling window, classifies
the driver's state, overlays everything on the live video, and logs every
frame's features to CSV for later ML training.

## How it works

```
webcam frame  ->  MediaPipe Face Mesh (468 landmarks)
              ->  EAR / MAR / PERCLOS features
              ->  ALERT / TIRED / MICROSLEEP state
              ->  on-screen overlay + per-frame CSV log
```

- **EAR (Eye Aspect Ratio)** — openness of each eye from 6 landmarks; the two
  eyes are averaged. Drops sharply when the eyes close.
- **MAR (Mouth Aspect Ratio)** — mouth openness from 8 landmarks; spikes on a
  yawn.
- **PERCLOS** — the percentage of frames in a rolling **30-second** window
  where EAR is below the closed-eye threshold (stored in a `deque`). This is
  the core sustained-drowsiness signal.
- **Auto-calibration** — for the first **10 seconds** the app records your
  open-eye EAR baseline and sets the closed threshold to **~75%** of it, so it
  adapts to each person's face and camera.

### States

| State        | Rule                                                        |
|--------------|-------------------------------------------------------------|
| `ALERT`      | default                                                     |
| `TIRED`      | PERCLOS above 15%, or a yawn detected (MAR high)            |
| `MICROSLEEP` | eyes closed continuously for more than **2 seconds**        |

Thresholds live in `src/detector.py` (`Thresholds` dataclass) and are easy to tune.

## Project structure

```
.
├── main.py              # webcam loop, overlay, CSV logging (OpenCV + MediaPipe)
├── src/
│   ├── features.py      # pure EAR / MAR / PERCLOS maths (no OpenCV)
│   └── detector.py      # landmark extraction + state classification
├── tests/
│   └── test_features.py # unit tests for the pure feature functions
├── requirements.txt
└── README.md
```

`src/features.py` has **no OpenCV/MediaPipe imports** — every function is a
pure transform, so the maths is unit-testable without a webcam.

## Setup

Requires **Python 3.10+**. MediaPipe ships wheels for 3.10–3.12; `pip` will
pick a compatible one. If install fails on a very new Python, use 3.11/3.12.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py                 # default webcam, logs to drowsiness_log.csv
python main.py --camera 1      # choose a different camera index
python main.py --no-log        # run without writing a CSV
python main.py --calibration 5 # shorter calibration window (seconds)
```

For the first ~10 seconds keep your eyes open and look at the camera so it can
calibrate. Press **`q`** or **Esc** to quit.

## CSV output

One row per frame (while a face is visible):

```
timestamp,ear,mar,perclos,state
2026-10-12T15:30:01.123456,0.3021,0.1104,0.0000,ALERT
...
```

This is the training data for later phases (e.g. an ML drowsiness classifier).

## Tests

```bash
python -m pytest            # if pytest is installed
python tests/test_features.py   # or run standalone, no pytest needed
```

## Notes & limitations (Phase 1)

- Single face only (`max_num_faces=1`), good frontal lighting assumed.
- Thresholds are heuristic defaults; tune `Thresholds` for your setup.
- Calibration assumes you are alert and eyes-open during the warm-up window.
- No audible alarm yet — state is visual + logged. That (and ML) is Phase 2+.
