"""
Optional extension only.
This is not needed for the first RP result.

It extracts lightweight Depth/RGB features when paths exist in sample_index.csv.
Depth uses .mat depth frames. RGB requires opencv-python and matching video files.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from config import INDEX_DIR, FEATURE_DIR
from utils_features import extract_depth_features, extract_rgb_features

def main():
    df = pd.read_csv(INDEX_DIR / "sample_index.csv")
    Xd, depth_ids = [], []
    Xr, rgb_ids = [], []

    for _, row in df.iterrows():
        sid = row["sample_id"]

        if isinstance(row.get("depth_path", ""), str) and row["depth_path"]:
            try:
                Xd.append(extract_depth_features(Path(row["depth_path"])))
                depth_ids.append(sid)
            except Exception as e:
                warnings.warn(f"Depth failed for {sid}: {e}")

        if isinstance(row.get("rgb_path", ""), str) and row["rgb_path"]:
            try:
                Xr.append(extract_rgb_features(Path(row["rgb_path"])))
                rgb_ids.append(sid)
            except ImportError:
                print("OpenCV missing. Install with: pip install opencv-python")
                break
            except Exception as e:
                warnings.warn(f"RGB failed for {sid}: {e}")

    if Xd:
        np.save(FEATURE_DIR / "X_depth_optional.npy", np.vstack(Xd))
        pd.DataFrame({"sample_id": depth_ids}).to_csv(FEATURE_DIR / "depth_optional_ids.csv", index=False)
        print("Saved optional depth features:", len(Xd))
    if Xr:
        np.save(FEATURE_DIR / "X_rgb_optional.npy", np.vstack(Xr))
        pd.DataFrame({"sample_id": rgb_ids}).to_csv(FEATURE_DIR / "rgb_optional_ids.csv", index=False)
        print("Saved optional RGB features:", len(Xr))

    if not Xd and not Xr:
        print("No optional depth/RGB features exported.")

if __name__ == "__main__":
    main()
