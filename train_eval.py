"""Train and evaluate the machine-learning models against threshold baselines.

Run from the project root (venv active), after prepare_yawn_features.py:

    python train_eval.py                  # both tasks
    python train_eval.py --tasks eye      # eye-state task only
    python train_eval.py --seeds 5        # fewer repeats (faster)

Two tasks, both with a threshold-rule baseline and ML models:

  yawn : whole-face images -> MediaPipe landmark features -> yawn / no_yawn
         baselines : the Phase-1 rule (MAR > 0.6) and a MAR threshold tuned on train
         ML        : Random Forest, Logistic Regression
  eye  : tight eye crops -> Open / Closed (positive class = Closed)
         baseline  : a contrast rule (grey-level spread in the eye centre), tuned on train
         ML        : Random Forest and Logistic Regression on HOG features

Everything the report needs is written to results/ (tables, figures) and the
trained models to models/.  Thresholds are tuned on the TRAIN split only; the
TEST split is used once per model for the numbers you report.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")  # write PNG files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
METRICS = ["accuracy", "precision", "recall", "f1"]
EYE_SIZE = 64
# HOG settings: window, block, stride, cell, bins -> 1764 values for a 64x64 image
HOG_PARAMS = ((EYE_SIZE, EYE_SIZE), (16, 16), (8, 8), (8, 8), 9)

# Same order as prepare_yawn_features.FEATURES (kept literal so this file stands alone)
YAWN_FEATURES = [
    "mar", "ear", "mouth_open_inner", "mouth_open_outer",
    "mouth_width", "mouth_inner_width", "chin_nose", "lowerlip_chin",
]
PHASE1_MAR = 0.6  # yawn threshold used by the live Phase-1 app (src/detector.py)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def score(y_true, y_pred) -> dict:
    """Accuracy plus precision / recall / F1 for the positive class (label 1)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def fit_threshold_rule(x: np.ndarray, y: np.ndarray):
    """Pick the single-feature threshold (and direction) with the best balanced
    accuracy on the training data.  direction=+1 means 'positive if x > t',
    direction=-1 means 'positive if x < t'."""
    candidates = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 197)))
    best_score, best_t, best_dir = -1.0, None, 1
    for direction in (1, -1):
        for t in candidates:
            pred = (x > t) if direction == 1 else (x < t)
            tpr = pred[y == 1].mean()
            tnr = (~pred[y == 0]).mean()
            balanced = (tpr + tnr) / 2.0
            if balanced > best_score:
                best_score, best_t, best_dir = balanced, float(t), direction
    return best_t, best_dir


def apply_rule(x: np.ndarray, t: float, direction: int) -> np.ndarray:
    return (x > t) if direction == 1 else (x < t)


def evaluate_ml(task: str, Xtr, ytr, Xte, yte, n_seeds: int):
    """Random Forest over several seeds + one Logistic Regression."""
    rows, keep = [], {}
    for seed in range(n_seeds):
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed
        )
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)
        rows.append({"task": task, "model": "Random Forest", "seed": seed, **score(yte, pred)})
        if seed == 0:
            keep["rf"], keep["rf_pred"] = rf, pred
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    pred = lr.predict(Xte)
    rows.append({"task": task, "model": "Logistic Regression", "seed": -1, **score(yte, pred)})
    keep["lr_pred"] = pred
    return rows, keep


def repeated_cv_rows(task: str, X, y, repeats: int):
    """Repeated stratified 5-fold CV of a Random Forest on train+test pooled."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=0)
    res = cross_validate(rf, X, y, cv=cv, scoring=METRICS)
    name = f"Random Forest (5-fold CV x{repeats}, pooled)"
    rows = []
    for i in range(len(res["test_accuracy"])):
        rows.append({"task": task, "model": name, "seed": i,
                     **{m: res[f"test_{m}"][i] for m in METRICS}})
    return rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def save_confusion(y_true, y_pred, class_names, title, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_comparison_bar(summary: pd.DataFrame, task: str, path: Path) -> None:
    df = summary[summary["task"] == task]
    labels = [m.replace(" (", "\n(") for m in df["model"]]
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(df)), 3.6))
    ax.bar(x - w / 2, df["recall_mean"], w, yerr=df["recall_std"].fillna(0), label="Recall", capsize=3)
    ax.bar(x + w / 2, df["f1_mean"], w, yerr=df["f1_std"].fillna(0), label="F1", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test split)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_importance(model, names, path: Path) -> None:
    order = np.argsort(model.feature_importances_)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(np.array(names)[order], model.feature_importances_[order])
    ax.set_xlabel("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_error_montage(imgs, y_true, y_pred, path: Path, max_items: int = 16) -> None:
    wrong = np.where(y_true != y_pred)[0][:max_items]
    if len(wrong) == 0:
        return
    names = {0: "Open", 1: "Closed"}
    cols = 4
    rows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(f"true {names[int(y_true[i])]}\npred {names[int(y_pred[i])]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def run_yawn(args, out: Path, models_dir: Path, all_rows: list, info: dict) -> None:
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"[yawn] {feat_path} not found - run prepare_yawn_features.py first. Skipping.")
        return
    df = pd.read_csv(feat_path)
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    if tr.empty or te.empty:
        print("[yawn] train or test split is empty. Skipping.")
        return
    Xtr, ytr = tr[YAWN_FEATURES].to_numpy(float), tr["label"].to_numpy(int)
    Xte, yte = te[YAWN_FEATURES].to_numpy(float), te["label"].to_numpy(int)
    names = ["no_yawn", "yawn"]
    print(f"[yawn] train {len(tr)} / test {len(te)} images with a detected face")

    # Baseline 1: the rule already used in the live app.
    pred = apply_rule(Xte[:, 0], PHASE1_MAR, 1)
    all_rows.append({"task": "yawn", "model": f"Phase-1 rule (MAR > {PHASE1_MAR})",
                     "seed": -1, **score(yte, pred)})
    save_confusion(yte, pred, names, "Phase-1 MAR rule", out / "cm_yawn_phase1_rule.png")

    # Baseline 2: the same single feature, threshold tuned on the training set.
    t, d = fit_threshold_rule(Xtr[:, 0], ytr)
    pred = apply_rule(Xte[:, 0], t, d)
    all_rows.append({"task": "yawn", "model": "Tuned MAR threshold", "seed": -1, **score(yte, pred)})
    save_confusion(yte, pred, names, "Tuned MAR threshold", out / "cm_yawn_tuned_rule.png")
    info["yawn_tuned_mar"] = {"threshold": t, "direction": d}
    print(f"[yawn] tuned MAR threshold: {'>' if d == 1 else '<'} {t:.3f}")

    rows, keep = evaluate_ml("yawn", Xtr, ytr, Xte, yte, args.seeds)
    all_rows += rows
    all_rows += repeated_cv_rows("yawn", np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]), args.cv_repeats)
    save_confusion(yte, keep["rf_pred"], names, "Random Forest (seed 0)", out / "cm_yawn_random_forest.png")
    save_importance(keep["rf"], YAWN_FEATURES, out / "importance_yawn.png")

    wrong = te.reset_index(drop=True)[keep["rf_pred"] != yte][["class", "file"]]
    wrong.to_csv(out / "yawn_errors_test.csv", index=False)

    joblib.dump({"model": keep["rf"], "features": YAWN_FEATURES, "task": "yawn",
                 "positive": "yawn"}, models_dir / "yawn_rf.joblib")


def load_eye_split(root: Path, split: str):
    imgs, labels, names = [], [], []
    for cls, label in (("Open", 0), ("Closed", 1)):
        folder = root / split / cls
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            imgs.append(cv2.resize(g, (EYE_SIZE, EYE_SIZE), interpolation=cv2.INTER_AREA))
            labels.append(label)
            names.append(f"{cls}/{p.name}")
    return np.array(imgs, dtype=np.uint8), np.array(labels, dtype=int), names


def hog_features(imgs: np.ndarray) -> np.ndarray:
    hog = cv2.HOGDescriptor(*HOG_PARAMS)
    return np.array([hog.compute(cv2.equalizeHist(im)).ravel() for im in imgs], dtype=np.float32)


def contrast_feature(imgs: np.ndarray) -> np.ndarray:
    """Grey-level spread in the centre of the crop (open eyes show iris against
    sclera, closed eyes are mostly uniform skin)."""
    return np.array([im[16:48, 8:56].std() for im in imgs], dtype=float)


def run_eye(args, out: Path, models_dir: Path, all_rows: list, info: dict) -> None:
    root = Path(args.data)
    imgs_tr, ytr, _ = load_eye_split(root, "train")
    imgs_te, yte, names_te = load_eye_split(root, "test")
    if len(ytr) == 0 or len(yte) == 0:
        print("[eye] no eye images found. Skipping.")
        return
    print(f"[eye] train {len(ytr)} / test {len(yte)} eye crops")
    class_names = ["Open", "Closed"]  # positive class = Closed

    xtr, xte = contrast_feature(imgs_tr), contrast_feature(imgs_te)
    t, d = fit_threshold_rule(xtr, ytr)
    pred = apply_rule(xte, t, d)
    all_rows.append({"task": "eye", "model": "Contrast rule (tuned)", "seed": -1, **score(yte, pred)})
    save_confusion(yte, pred, class_names, "Contrast rule", out / "cm_eye_contrast_rule.png")
    info["eye_contrast_rule"] = {"threshold": t, "direction": d}
    print(f"[eye] contrast rule: closed if spread {'>' if d == 1 else '<'} {t:.2f}")

    Htr, Hte = hog_features(imgs_tr), hog_features(imgs_te)
    rows, keep = evaluate_ml("eye", Htr, ytr, Hte, yte, args.seeds)
    all_rows += rows
    all_rows += repeated_cv_rows("eye", np.vstack([Htr, Hte]), np.concatenate([ytr, yte]), args.cv_repeats)
    save_confusion(yte, keep["rf_pred"], class_names, "Random Forest + HOG (seed 0)",
                   out / "cm_eye_random_forest.png")
    save_error_montage(imgs_te, yte, keep["rf_pred"], out / "eye_errors_montage.png")

    wrong = [(names_te[i], class_names[yte[i]], class_names[int(keep["rf_pred"][i])])
             for i in np.where(keep["rf_pred"] != yte)[0]]
    pd.DataFrame(wrong, columns=["file", "actual", "predicted"]).to_csv(
        out / "eye_errors_test.csv", index=False)

    joblib.dump({"model": keep["rf"], "img_size": EYE_SIZE, "hog_params": HOG_PARAMS,
                 "task": "eye_state", "positive": "closed"}, models_dir / "eye_rf.joblib")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model), g in per_run.groupby(["task", "model"], sort=False):
        row = {"task": task, "model": model, "n_runs": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f}" if (std is None or np.isnan(std)) else f"{mean:.3f} ± {std:.3f}"


def write_markdown(summary: pd.DataFrame, info: dict, args, path: Path) -> None:
    lines = ["# Results summary (test split unless stated)", ""]
    titles = {"yawn": "Yawn detection (positive class = yawn)",
              "eye": "Eye state (positive class = closed)"}
    for task in summary["task"].unique():
        lines += [f"## {titles.get(task, task)}", "",
                  "| Model | Runs | Accuracy | Precision | Recall | F1 |",
                  "|---|---|---|---|---|---|"]
        for _, r in summary[summary["task"] == task].iterrows():
            lines.append(
                f"| {r['model']} | {int(r['n_runs'])} | "
                + " | ".join(fmt(r[f"{m}_mean"], r[f"{m}_std"]) for m in METRICS) + " |")
        lines.append("")
    lines += ["## Thresholds learned on the training split", "", "```", json.dumps(info, indent=2), "```", "",
              "## Notes for the critical discussion", "",
              "- Rows with 'Runs' > 1 show mean ± standard deviation over repeated runs "
              f"(Random Forest seeds 0-{args.seeds - 1}, or CV folds).",
              "- Rules and Logistic Regression are deterministic, so they have a single run.",
              "- The train/test split comes from the dataset authors. It is not known whether the "
              "same people appear on both sides, so scores may be optimistic (possible leakage).",
              "- 'Pooled CV' mixes train and test images, so it is even more likely to be optimistic.",
              "- Eye crops are tight crops from the dataset; the live app would crop eyes from "
              "webcam frames with landmarks, so a domain shift is likely.",
              "- Yawn images where MediaPipe found no face are excluded; see data/features/yawn_failures.csv.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_summary(args, out: Path) -> None:
    lines = ["# Dataset summary", ""]
    root = Path(args.data)
    lines += ["## Images on disk", "", "| Split | Class | Images |", "|---|---|---|"]
    for split in ("train", "test"):
        for cls in ("Open", "Closed", "no_yawn", "yawn"):
            folder = root / split / cls
            n = sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.is_dir() else 0
            lines.append(f"| {split} | {cls} | {n} |")
    feat_path, fail_path = Path(args.features), Path(args.features).with_name("yawn_failures.csv")
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        fails = pd.read_csv(fail_path) if fail_path.exists() else pd.DataFrame(columns=["split", "class"])
        lines += ["", "## Yawn images with a detected face", "",
                  "| Split | Class | Face found | No face | Detection rate |", "|---|---|---|---|---|"]
        for split in ("train", "test"):
            for cls in ("no_yawn", "yawn"):
                ok = int(((df["split"] == split) & (df["class"] == cls)).sum())
                bad = int(((fails["split"] == split) & (fails["class"] == cls)).sum())
                rate = ok / (ok + bad) if ok + bad else float("nan")
                lines.append(f"| {split} | {cls} | {ok} | {bad} | {rate:.1%} |")
    (out / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Train and evaluate drowsiness-cue models")
    p.add_argument("--data", default="data/YawnEye")
    p.add_argument("--features", default="data/features/yawn_features.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--models", default="models")
    p.add_argument("--tasks", default="yawn,eye", help="comma separated: yawn,eye")
    p.add_argument("--seeds", type=int, default=10, help="Random Forest repeats")
    p.add_argument("--cv-repeats", type=int, default=3, help="repeats of 5-fold CV")
    args = p.parse_args()

    out, models_dir = Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    all_rows: list = []
    info: dict = {}
    if "yawn" in tasks:
        run_yawn(args, out, models_dir, all_rows, info)
    if "eye" in tasks:
        run_eye(args, out, models_dir, all_rows, info)
    if not all_rows:
        raise SystemExit("Nothing was evaluated - check the paths and --tasks.")

    per_run = pd.DataFrame(all_rows)
    per_run.to_csv(out / "metrics_per_run.csv", index=False)
    summary = build_summary(per_run)
    summary.to_csv(out / "summary.csv", index=False)
    write_markdown(summary, info, args, out / "summary.md")
    write_dataset_summary(args, out)
    (models_dir / "baselines.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    for task in summary["task"].unique():
        save_comparison_bar(summary, task, out / f"comparison_{task}.png")

    print("\n" + (out / "summary.md").read_text(encoding="utf-8"))
    print(f"Files written to {out}/ and {models_dir}/")


if __name__ == "__main__":
    main()
