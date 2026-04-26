"""
07_train_extended_modalities.py

Clean-condition comparison for optional UTD-MHAD modalities.
Run after:
  01_scan_dataset.py
  02_extract_features.py
  06_optional_extract_depth_rgb_features.py

This script does NOT replace the core Skeleton+Inertial experiment.
It only checks whether lightweight Depth/RGB features can be used as
additional modalities under clean sensing conditions.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import FEATURE_DIR, RESULT_DIR, TRAIN_SUBJECTS, TEST_SUBJECTS, RANDOM_SEEDS
from utils_models import make_model, evaluate_model, summarize_results


def _load_required_features():
    X = {
        "Skeleton": np.load(FEATURE_DIR / "X_skeleton.npy"),
        "Inertial": np.load(FEATURE_DIR / "X_inertial.npy"),
    }
    y = np.load(FEATURE_DIR / "y.npy")
    subjects = np.load(FEATURE_DIR / "subjects.npy")
    sample_ids = pd.read_csv(FEATURE_DIR / "sample_ids.csv")["sample_id"].astype(str).tolist()
    return X, y, subjects, sample_ids


def _try_load_optional(name, feature_file, ids_file, master_ids):
    """Load optional feature array and align it to master_ids if available."""
    feature_path = FEATURE_DIR / feature_file
    ids_path = FEATURE_DIR / ids_file

    if not feature_path.exists() or not ids_path.exists():
        print(f"[INFO] Optional modality missing: {name} ({feature_file}, {ids_file})")
        return None

    X_opt = np.load(feature_path)
    opt_ids = pd.read_csv(ids_path)["sample_id"].astype(str).tolist()

    if len(X_opt) != len(opt_ids):
        raise ValueError(f"{name}: feature rows ({len(X_opt)}) != id rows ({len(opt_ids)})")

    id_to_idx = {sid: i for i, sid in enumerate(opt_ids)}
    missing = [sid for sid in master_ids if sid not in id_to_idx]
    if missing:
        print(f"[WARN] {name}: {len(missing)} master samples missing optional features. Skipping this modality.")
        return None

    aligned = np.vstack([X_opt[id_to_idx[sid]] for sid in master_ids])
    print(f"[OK] Loaded optional modality: {name}, shape={aligned.shape}")
    return aligned


def _make_setting_dict(X):
    settings = {
        "Skeleton-only": ["Skeleton"],
        "Inertial-only": ["Inertial"],
        "Skeleton+Inertial": ["Skeleton", "Inertial"],
    }

    if "Depth" in X:
        settings["Depth-only"] = ["Depth"]
        settings["Skeleton+Inertial+Depth"] = ["Skeleton", "Inertial", "Depth"]

    if "RGB" in X:
        settings["RGB-only"] = ["RGB"]
        settings["Skeleton+Inertial+RGB"] = ["Skeleton", "Inertial", "RGB"]

    if "Depth" in X and "RGB" in X:
        settings["Depth+RGB"] = ["Depth", "RGB"]
        settings["All modalities"] = ["Skeleton", "Inertial", "Depth", "RGB"]

    return settings


def _concat_modalities(X, modality_names):
    return np.hstack([X[m] for m in modality_names])


def main():
    X, y, subjects, sample_ids = _load_required_features()

    X_depth = _try_load_optional("Depth", "X_depth_optional.npy", "depth_optional_ids.csv", sample_ids)
    if X_depth is not None:
        X["Depth"] = X_depth

    X_rgb = _try_load_optional("RGB", "X_rgb_optional.npy", "rgb_optional_ids.csv", sample_ids)
    if X_rgb is not None:
        X["RGB"] = X_rgb

    train_mask = np.isin(subjects, list(TRAIN_SUBJECTS))
    test_mask = np.isin(subjects, list(TEST_SUBJECTS))
    y_train, y_test = y[train_mask], y[test_mask]

    settings = _make_setting_dict(X)

    print("\nClean-condition extended modality comparison")
    print("Train samples:", len(y_train), "Test samples:", len(y_test))
    print("Available modalities:", {k: v.shape for k, v in X.items()})
    print("Settings:")
    for name, mods in settings.items():
        print(f"  - {name}: {mods}")

    rows = []
    for seed in RANDOM_SEEDS:
        for setting_name, mods in settings.items():
            X_all = _concat_modalities(X, mods)
            X_train, X_test = X_all[train_mask], X_all[test_mask]

            model = make_model(seed)
            model.fit(X_train, y_train)
            rows.append({
                "seed": seed,
                "setting": setting_name,
                "modalities": "+".join(mods),
                "n_features": X_all.shape[1],
                **evaluate_model(model, X_test, y_test),
            })

    raw = pd.DataFrame(rows)
    summary = summarize_results(raw, ["setting", "modalities", "n_features"])
    summary = summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)

    raw.to_csv(RESULT_DIR / "extended_clean_results_by_seed.csv", index=False)
    summary.to_csv(RESULT_DIR / "extended_clean_results_summary.csv", index=False)

    # Export a copy in LaTeX-friendly form.
    latex = summary.copy()
    latex["Accuracy"] = latex.apply(lambda r: f"{r['accuracy_mean']:.3f} $\\pm$ {r['accuracy_std']:.3f}", axis=1)
    latex["Macro-F1"] = latex.apply(lambda r: f"{r['macro_f1_mean']:.3f} $\\pm$ {r['macro_f1_std']:.3f}", axis=1)
    latex_out = latex[["setting", "modalities", "n_features", "Accuracy", "Macro-F1"]]
    latex_out = latex_out.rename(columns={"setting": "Setting", "modalities": "Modalities", "n_features": "Features"})
    (RESULT_DIR / "table_extended_clean_latex.txt").write_text(latex_out.to_latex(index=False, escape=False), encoding="utf-8")

    print("\nExtended clean-condition summary:")
    print(summary)
    print("\nSaved:")
    print(" -", RESULT_DIR / "extended_clean_results_by_seed.csv")
    print(" -", RESULT_DIR / "extended_clean_results_summary.csv")
    print(" -", RESULT_DIR / "table_extended_clean_latex.txt")


if __name__ == "__main__":
    main()
