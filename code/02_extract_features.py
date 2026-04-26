from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from config import INDEX_DIR, FEATURE_DIR, ENABLE_DEPTH_FEATURES, ENABLE_RGB_FEATURES
from utils_features import extract_skeleton_features, extract_inertial_features, extract_depth_features, extract_rgb_features

def main():
    index_path = INDEX_DIR / "sample_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("Run 01_scan_dataset.py first.")

    df = pd.read_csv(index_path)

    Xs, Xi, Xd, Xr = [], [], [], []
    y, subjects, trials, sample_ids = [], [], [], []

    for _, row in df.iterrows():
        sid = row["sample_id"]
        try:
            Xs.append(extract_skeleton_features(Path(row["skeleton_path"])))
            Xi.append(extract_inertial_features(Path(row["inertial_path"])))

            if ENABLE_DEPTH_FEATURES and isinstance(row.get("depth_path", ""), str) and row["depth_path"]:
                Xd.append(extract_depth_features(Path(row["depth_path"])))

            if ENABLE_RGB_FEATURES and isinstance(row.get("rgb_path", ""), str) and row["rgb_path"]:
                Xr.append(extract_rgb_features(Path(row["rgb_path"])))

            y.append(int(row["action"]))
            subjects.append(int(row["subject"]))
            trials.append(int(row["trial"]))
            sample_ids.append(sid)

        except Exception as e:
            warnings.warn(f"Skipping {sid}: {e}")

    Xs, Xi = np.vstack(Xs), np.vstack(Xi)
    y = np.array(y, dtype=np.int64)
    subjects = np.array(subjects, dtype=np.int64)
    trials = np.array(trials, dtype=np.int64)

    np.save(FEATURE_DIR / "X_skeleton.npy", Xs)
    np.save(FEATURE_DIR / "X_inertial.npy", Xi)
    np.save(FEATURE_DIR / "y.npy", y)
    np.save(FEATURE_DIR / "subjects.npy", subjects)
    np.save(FEATURE_DIR / "trials.npy", trials)
    pd.DataFrame({"sample_id": sample_ids}).to_csv(FEATURE_DIR / "sample_ids.csv", index=False)

    if ENABLE_DEPTH_FEATURES and Xd:
        np.save(FEATURE_DIR / "X_depth_optional.npy", np.vstack(Xd))
    if ENABLE_RGB_FEATURES and Xr:
        np.save(FEATURE_DIR / "X_rgb_optional.npy", np.vstack(Xr))

    print("Feature extraction complete.")
    print("Samples:", len(y))
    print("Skeleton feature shape:", Xs.shape)
    print("Inertial feature shape:", Xi.shape)
    print("Classes:", len(np.unique(y)))
    print("Subjects:", sorted(np.unique(subjects)))

if __name__ == "__main__":
    main()
