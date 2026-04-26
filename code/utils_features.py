from pathlib import Path
import numpy as np
from utils_io import load_mat_var

def safe_stats_1d(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(10, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array([
        np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
        np.percentile(x, 25), np.percentile(x, 75),
        np.sqrt(np.mean(x ** 2)), np.mean(np.abs(x)),
        np.sum(x ** 2) / max(len(x), 1),
    ], dtype=np.float64)

def time_series_features(ts):
    ts = np.asarray(ts, dtype=np.float64)
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    if ts.ndim == 1:
        ts = ts[:, None]

    feats = []
    for d in range(ts.shape[1]):
        feats.append(safe_stats_1d(ts[:, d]))

    diff = np.diff(ts, axis=0) if ts.shape[0] >= 2 else np.zeros_like(ts)
    for d in range(diff.shape[1]):
        xd = diff[:, d]
        feats.append(np.array([
            np.mean(xd), np.std(xd),
            np.sqrt(np.mean(xd ** 2)), np.mean(np.abs(xd))
        ], dtype=np.float64))
    return np.concatenate(feats)

def extract_skeleton_features(path: Path):
    # d_skel: (20 joints, 3 coordinates, T frames)
    arr = np.asarray(load_mat_var(path, "d_skel"), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected skeleton shape {arr.shape} in {path}")
    skel = np.transpose(arr, (2, 0, 1))  # (T, 20, 3)
    skel = skel - np.mean(skel, axis=1, keepdims=True)  # center per frame
    return time_series_features(skel.reshape(skel.shape[0], -1))

def extract_inertial_features(path: Path):
    # d_iner: (T, 6)
    arr = np.asarray(load_mat_var(path, "d_iner"), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected inertial shape {arr.shape} in {path}")
    return time_series_features(arr)

def extract_depth_features(path: Path):
    # Optional; d_depth: (H, W, T). Keep it lightweight.
    arr = np.asarray(load_mat_var(path, "d_depth"), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected depth shape {arr.shape} in {path}")
    H, W, T = arr.shape
    rows = []
    for t in range(T):
        frame = arr[:, :, t]
        nz = frame[frame > 0]
        if nz.size == 0:
            rows.append([0, 0, 0, 0, 0, 0])
        else:
            rows.append([nz.mean(), nz.std(), nz.min(), nz.max(), np.median(nz), nz.size / (H * W)])
    return time_series_features(np.asarray(rows))

def extract_rgb_features(path: Path, max_frames=48, resize_hw=(64, 64)):
    # Optional; requires opencv-python.
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    rows, count = [], 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, resize_hw)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64)
        rows.append([
            frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean(),
            frame[:, :, 0].std(), frame[:, :, 1].std(), frame[:, :, 2].std(),
        ])
        count += 1
    cap.release()
    if not rows:
        return np.zeros(84, dtype=np.float64)
    return time_series_features(np.asarray(rows))

def degrade_features(X, rate, rng, noise_scale=0.05):
    X = np.asarray(X, dtype=np.float64)
    if rate <= 0:
        return X.copy()
    Xd = X.copy()
    std = np.std(Xd, axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xd = Xd + rng.normal(0, noise_scale * rate, size=Xd.shape) * std
    mask = rng.random(size=Xd.shape) < rate
    Xd[mask] = 0.0
    return Xd

def build_reliability_features(Xs, Xi, skel_rate, iner_rate):
    r_s, r_i = 1.0 - skel_rate, 1.0 - iner_rate
    indicators = np.tile(np.array([r_s, r_i, skel_rate, iner_rate], dtype=np.float64), (Xs.shape[0], 1))
    return np.hstack([r_s * Xs, r_i * Xi, indicators])
