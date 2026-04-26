import numpy as np
import pandas as pd

from config import FEATURE_DIR, RESULT_DIR, TRAIN_SUBJECTS, TEST_SUBJECTS, RANDOM_SEEDS, DEGRADATION_RATES
from utils_features import degrade_features, build_reliability_features
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

    rows = []
    for seed in RANDOM_SEEDS:
        rng_train = np.random.default_rng(seed)

        m_skel = make_model(seed); m_skel.fit(Xs_train, y_train)
        m_iner = make_model(seed); m_iner.fit(Xi_train, y_train)
        m_early = make_model(seed); m_early.fit(np.hstack([Xs_train, Xi_train]), y_train)

        # Train reliability-aware model with degraded augmentation.
        X_aug, y_aug = [], []
        for rate in DEGRADATION_RATES:
            Xs_d = degrade_features(Xs_train, rate, rng_train)
            Xi_d = degrade_features(Xi_train, rate, rng_train)
            X_aug.append(build_reliability_features(Xs_d, Xi_d, rate, rate))
            y_aug.append(y_train)
        m_ra = make_model(seed)
        m_ra.fit(np.vstack(X_aug), np.concatenate(y_aug))

        for rate in DEGRADATION_RATES:
            rng_test = np.random.default_rng(seed + int(rate * 1000) + 7)
            Xs_d = degrade_features(Xs_test, rate, rng_test)
            Xi_d = degrade_features(Xi_test, rate, rng_test)

            rows.append({"seed": seed, "degradation_rate": rate, "model": "Skeleton-only",
                         **evaluate_model(m_skel, Xs_d, y_test)})
            rows.append({"seed": seed, "degradation_rate": rate, "model": "Inertial-only",
                         **evaluate_model(m_iner, Xi_d, y_test)})
            rows.append({"seed": seed, "degradation_rate": rate, "model": "Early fusion",
                         **evaluate_model(m_early, np.hstack([Xs_d, Xi_d]), y_test)})
            rows.append({"seed": seed, "degradation_rate": rate, "model": "Reliability-aware fusion",
                         **evaluate_model(m_ra, build_reliability_features(Xs_d, Xi_d, rate, rate), y_test)})

    raw = pd.DataFrame(rows)
    summary = summarize_results(raw, ["degradation_rate", "model"])
    raw.to_csv(RESULT_DIR / "degraded_results_by_seed.csv", index=False)
    summary.to_csv(RESULT_DIR / "degraded_results_summary.csv", index=False)

    print("\nDegraded-sensing summary:")
    print(summary)

if __name__ == "__main__":
    main()
