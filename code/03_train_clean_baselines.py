import numpy as np
import pandas as pd

from config import FEATURE_DIR, RESULT_DIR, TRAIN_SUBJECTS, TEST_SUBJECTS, RANDOM_SEEDS
from utils_features import build_reliability_features
from utils_models import make_model, evaluate_model, summarize_results

def main():
    Xs = np.load(FEATURE_DIR / "X_skeleton.npy")
    Xi = np.load(FEATURE_DIR / "X_inertial.npy")
    y = np.load(FEATURE_DIR / "y.npy")
    subjects = np.load(FEATURE_DIR / "subjects.npy")

    train_mask = np.isin(subjects, list(TRAIN_SUBJECTS))
    test_mask = np.isin(subjects, list(TEST_SUBJECTS))

    Xs_train, Xs_test = Xs[train_mask], Xs[test_mask]
    Xi_train, Xi_test = Xi[train_mask], Xi[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train samples:", len(y_train), "Test samples:", len(y_test))

    rows = []
    for seed in RANDOM_SEEDS:
        for model_name, Xtr, Xte in [
            ("Skeleton-only", Xs_train, Xs_test),
            ("Inertial-only", Xi_train, Xi_test),
            ("Early fusion", np.hstack([Xs_train, Xi_train]), np.hstack([Xs_test, Xi_test])),
            ("Reliability-aware fusion", build_reliability_features(Xs_train, Xi_train, 0.0, 0.0),
             build_reliability_features(Xs_test, Xi_test, 0.0, 0.0)),
        ]:
            m = make_model(seed)
            m.fit(Xtr, y_train)
            rows.append({"seed": seed, "model": model_name, **evaluate_model(m, Xte, y_test)})

    raw = pd.DataFrame(rows)
    summary = summarize_results(raw, ["model"])
    raw.to_csv(RESULT_DIR / "clean_results_by_seed.csv", index=False)
    summary.to_csv(RESULT_DIR / "clean_results_summary.csv", index=False)

    print("\nClean-condition summary:")
    print(summary)

if __name__ == "__main__":
    main()
