"""Turn the Kaggle Yawn-Eye face images (yawn / no_yawn) into a feature table.

Run from the project root with the virtual environment active:

    python prepare_yawn_features.py --limit 20     # quick smoke test first
    python prepare_yawn_features.py                # full run (a few minutes)

Reads : data/YawnEye/{train,test}/{yawn,no_yawn}/*.jpg
Writes: data/features/yawn_features.csv   one row per image where a face was found
        data/features/yawn_failures.csv   images where no face was detected

The EAR / MAR maths is reused from src/ so the offline experiments use exactly
the same landmark indices and formulas as the live webcam app.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Sequence

import cv2
import mediapipe as mp

from src.detector import DrowsinessDetector
from src.features import euclidean_distance

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = {"no_yawn": 0, "yawn": 1}  # folder name -> label (1 = yawn)

# Column order shared with train_eval.py
FEATURES = [
    "mar",                # mouth aspect ratio from src.features (Phase 1 yawn cue)
    "ear",                # mean eye aspect ratio (eyes narrow during a yawn)
    "mouth_open_inner",   # inner-lip gap / face height
    "mouth_open_outer",   # outer-lip gap / face height
    "mouth_width",        # outer mouth corner distance / face width
    "mouth_inner_width",  # inner mouth corner distance / face width
    "chin_nose",          # nose tip to chin / face height (jaw drop)
    "lowerlip_chin",      # lower lip to chin / face height
]


def landmark_features(
    landmarks: Sequence, width: int, height: int, detector: DrowsinessDetector
) -> Dict[str, float]:
    """Compute the feature dictionary for one face from its Face Mesh landmarks.

    Pure given its inputs (no camera, no MediaPipe call), so it can be tested
    with stand-in landmark objects that expose ``.x`` and ``.y`` in 0..1.
    """
    ear, mar = detector.compute_raw_features(landmarks, width, height)

    def pt(i: int):
        return (landmarks[i].x * width, landmarks[i].y * height)

    # Face size, used to make mouth measurements independent of image size.
    face_w = euclidean_distance(pt(234), pt(454))  # left / right face edge
    face_h = euclidean_distance(pt(10), pt(152))   # forehead top / chin
    if face_w <= 0 or face_h <= 0:
        raise ValueError("degenerate face box")

    return {
        "mar": mar,
        "ear": ear,
        "mouth_open_inner": euclidean_distance(pt(13), pt(14)) / face_h,
        "mouth_open_outer": euclidean_distance(pt(0), pt(17)) / face_h,
        "mouth_width": euclidean_distance(pt(61), pt(291)) / face_w,
        "mouth_inner_width": euclidean_distance(pt(78), pt(308)) / face_w,
        "chin_nose": euclidean_distance(pt(1), pt(152)) / face_h,
        "lowerlip_chin": euclidean_distance(pt(17), pt(152)) / face_h,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract yawn features from face images")
    p.add_argument("--data", default="data/YawnEye", help="dataset folder")
    p.add_argument("--out", default="data/features", help="output folder")
    p.add_argument("--limit", type=int, default=0,
                   help="only process the first N images per class (0 = all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = DrowsinessDetector()  # used only for its pure EAR/MAR helper
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,      # single images, not a video stream
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    rows, failures = [], []
    start = time.time()
    try:
        for split in ("train", "test"):
            for cls, label in CLASSES.items():
                folder = root / split / cls
                if not folder.is_dir():
                    raise SystemExit(f"Missing folder: {folder}")
                files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
                if args.limit:
                    files = files[: args.limit]
                found = 0
                for n, path in enumerate(files, 1):
                    img = cv2.imread(str(path))
                    if img is None:
                        failures.append((split, cls, path.name, "unreadable"))
                        continue
                    height, width = img.shape[:2]
                    result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if not result.multi_face_landmarks:
                        failures.append((split, cls, path.name, "no_face"))
                        continue
                    feats = landmark_features(
                        result.multi_face_landmarks[0].landmark, width, height, detector
                    )
                    rows.append({"split": split, "class": cls, "label": label,
                                 "file": path.name, **feats})
                    found += 1
                    if n % 200 == 0:
                        print(f"  {split}/{cls}: {n}/{len(files)}")
                print(f"{split:5s} {cls:8s} faces found: {found}/{len(files)}")
    finally:
        face_mesh.close()

    if not rows:
        raise SystemExit("No faces were found in any image - check the data folder.")

    cols = ["split", "class", "label", "file"] + FEATURES
    feat_path = out_dir / "yawn_features.csv"
    with open(feat_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    fail_path = out_dir / "yawn_failures.csv"
    with open(fail_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "class", "file", "reason"])
        writer.writerows(failures)

    print(f"\nSaved {len(rows)} rows to {feat_path}")
    print(f"Saved {len(failures)} failures to {fail_path}")
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
