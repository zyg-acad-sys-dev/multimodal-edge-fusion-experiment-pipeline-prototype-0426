"""
08_eval_extended_degraded_robustness.py

Degraded-sensing robustness for core and optional multimodal fusion settings.
Run after:
  02_extract_features.py
  06_optional_extract_depth_rgb_features.py
  07_train_extended_modalities.py   # optional but recommended

This script compares:
  - Early fusion: clean training only
  - Augmented early fusion: trained with degraded augmentation, no reliability indicators
  - Reliability-aware fusion: trained with degraded augmentation + reliability indicators

It runs these strategies for:
  - Core modalities: Skeleton + Inertial
  - All available modalities: Skeleton + Inertial + optional Depth + optional RGB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FEATURE_DIR, RESULT_DIR, FIGURE_DIR, TRAIN_SUBJECTS, TEST_SUBJECTS, RANDOM_SEEDS, DEGRADATION_RATES
from utils_features import degrade_features
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
    feature_path = FEATURE_DIR / feature_file
    ids_path = FEATURE_DIR / ids_file
    if not feature_path.exists() or not ids_path.exists():
        print(f"[INFO] Optional modality missing: {name}")
        return None

    X_opt = np.load(feature_path)
    opt_ids = pd.read_csv(ids_path)["sample_id"].astype(str).tolist()
    id_to_idx = {sid: i for i, sid in enumerate(opt_ids)}

    missing = [sid for sid in master_ids if sid not in id_to_idx]
    if missing:
        print(f"[WARN] {name}: {len(missing)} master samples missing optional features. Skipping this modality.")
        return None

    aligned = np.vstack([X_opt[id_to_idx[sid]] for sid in master_ids])
    print(f"[OK] Loaded optional modality: {name}, shape={aligned.shape}")
    return aligned


def _concat_feature_dict(X_dict, modality_names):
    return np.hstack([X_dict[m] for m in modality_names])


def _degrade_dict(X_dict, modality_names, rate, rng):
    return {m: degrade_features(X_dict[m], rate, rng) for m in modality_names}


def _build_reliability_multi(X_dict, modality_names, rate):
    """
    Generic reliability-aware representation:
    [r1*X1, r2*X2, ..., rM*XM, r_vector, degradation_vector]

    In this preliminary experiment all modalities share the same simulated
    degradation rate. The formula is kept generic so that future work can use
    modality-specific missingness, delay, or noise estimates.
    """
    weighted = []
    r_values = []
    d_values = []

    for m in modality_names:
        r = 1.0 - rate
        weighted.append(r * X_dict[m])
        r_values.append(r)
        d_values.append(rate)

    n = next(iter(X_dict.values())).shape[0]
    indicators = np.tile(np.array(r_values + d_values, dtype=np.float64), (n, 1))
    return np.hstack(weighted + [indicators])


def _train_augmented_early(X_train_dict, modality_names, y_train, seed):
    rng = np.random.default_rng(seed)
    X_aug, y_aug = [], []
    for rate in DEGRADATION_RATES:
        Xd = _degrade_dict(X_train_dict, modality_names, rate, rng)
        X_aug.append(_concat_feature_dict(Xd, modality_names))
        y_aug.append(y_train)
    model = make_model(seed)
    model.fit(np.vstack(X_aug), np.concatenate(y_aug))
    return model


def _train_reliability_aware(X_train_dict, modality_names, y_train, seed):
    rng = np.random.default_rng(seed)
    X_aug, y_aug = [], []
    for rate in DEGRADATION_RATES:
        Xd = _degrade_dict(X_train_dict, modality_names, rate, rng)
        X_aug.append(_build_reliability_multi(Xd, modality_names, rate))
        y_aug.append(y_train)
    model = make_model(seed)
    model.fit(np.vstack(X_aug), np.concatenate(y_aug))
    return model


def _make_modality_groups(X):
    groups = {
        "Core S+I": ["Skeleton", "Inertial"],
    }
    all_mods = ["Skeleton", "Inertial"]
    if "Depth" in X:
        all_mods.append("Depth")
    if "RGB" in X:
        all_mods.append("RGB")
    if len(all_mods) > 2:
        groups["All available"] = all_mods
    return groups


def _plot_extended_robustness(summary):
    if summary.empty:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Main figure: compare three strategy types for each modality group.
    for group_name in summary["modality_group"].unique():
        sub_group = summary[summary["modality_group"] == group_name]
        plt.figure(figsize=(6.8, 4.4))
        for strategy in ["Early fusion", "Augmented early fusion", "Reliability-aware fusion"]:
            sub = sub_group[sub_group["strategy"] == strategy].sort_values("degradation_rate")
            if sub.empty:
                continue
            x = sub["degradation_rate"].values * 100
            y = sub["macro_f1_mean"].values
            s = sub["macro_f1_std"].fillna(0).values
            plt.plot(x, y, marker="o", label=strategy)
            plt.fill_between(x, y - s, y + s, alpha=0.15)
        plt.xlabel("Simulated sensing degradation rate (%)")
        plt.ylabel("Macro-F1")
        plt.title(f"Extended robustness: {group_name}")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend(fontsize=8)
        plt.tight_layout()

        safe_name = group_name.lower().replace("+", "plus").replace(" ", "_")
        plt.savefig(FIGURE_DIR / f"extended_robustness_{safe_name}.pdf")
        plt.savefig(FIGURE_DIR / f"extended_robustness_{safe_name}.png", dpi=300)
        plt.close()

    # Compact comparison: reliability-aware core vs all available if available.
    plt.figure(figsize=(6.8, 4.4))
    for group_name in summary["modality_group"].unique():
        sub = summary[(summary["modality_group"] == group_name) & (summary["strategy"] == "Reliability-aware fusion")]
        sub = sub.sort_values("degradation_rate")
        if sub.empty:
            continue
        x = sub["degradation_rate"].values * 100
        y = sub["macro_f1_mean"].values
        s = sub["macro_f1_std"].fillna(0).values
        plt.plot(x, y, marker="o", label=f"{group_name}: reliability-aware")
        plt.fill_between(x, y - s, y + s, alpha=0.15)
    plt.xlabel("Simulated sensing degradation rate (%)")
    plt.ylabel("Macro-F1")
    plt.title("Reliability-aware fusion across modality groups")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "extended_reliability_comparison.pdf")
    plt.savefig(FIGURE_DIR / "extended_reliability_comparison.png", dpi=300)
    plt.close()


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

    modality_groups = _make_modality_groups(X)
    print("\nExtended degraded-sensing robustness")
    print("Available modalities:", {k: v.shape for k, v in X.items()})
    print("Modality groups:")
    for name, mods in modality_groups.items():
        print(f"  - {name}: {mods}")

    rows = []
    for group_name, mods in modality_groups.items():
        X_train_dict = {m: X[m][train_mask] for m in mods}
        X_test_dict = {m: X[m][test_mask] for m in mods}

        for seed in RANDOM_SEEDS:
            # Strategy 1: conventional early fusion, trained on clean features.
            m_early = make_model(seed)
            m_early.fit(_concat_feature_dict(X_train_dict, mods), y_train)

            # Strategy 2: augmentation only. This separates augmentation from reliability indicators.
            m_aug = _train_augmented_early(X_train_dict, mods, y_train, seed)

            # Strategy 3: degradation augmentation + reliability indicators.
            m_rel = _train_reliability_aware(X_train_dict, mods, y_train, seed)

            for rate in DEGRADATION_RATES:
                rng_test = np.random.default_rng(seed + int(rate * 1000) + 17)
                Xd_test = _degrade_dict(X_test_dict, mods, rate, rng_test)

                rows.append({
                    "seed": seed,
                    "modality_group": group_name,
                    "modalities": "+".join(mods),
                    "strategy": "Early fusion",
                    "degradation_rate": rate,
                    **evaluate_model(m_early, _concat_feature_dict(Xd_test, mods), y_test),
                })

                rows.append({
                    "seed": seed,
                    "modality_group": group_name,
                    "modalities": "+".join(mods),
                    "strategy": "Augmented early fusion",
                    "degradation_rate": rate,
                    **evaluate_model(m_aug, _concat_feature_dict(Xd_test, mods), y_test),
                })

                rows.append({
                    "seed": seed,
                    "modality_group": group_name,
                    "modalities": "+".join(mods),
                    "strategy": "Reliability-aware fusion",
                    "degradation_rate": rate,
                    **evaluate_model(m_rel, _build_reliability_multi(Xd_test, mods, rate), y_test),
                })

    raw = pd.DataFrame(rows)
    summary = summarize_results(raw, ["modality_group", "modalities", "strategy", "degradation_rate"])
    summary = summary.sort_values(["modality_group", "strategy", "degradation_rate"]).reset_index(drop=True)

    raw.to_csv(RESULT_DIR / "extended_degraded_results_by_seed.csv", index=False)
    summary.to_csv(RESULT_DIR / "extended_degraded_results_summary.csv", index=False)

    # LaTeX-friendly table.
    latex = summary.copy()
    latex["Degradation (%)"] = (latex["degradation_rate"] * 100).astype(int)
    latex["Accuracy"] = latex.apply(lambda r: f"{r['accuracy_mean']:.3f} $\\pm$ {r['accuracy_std']:.3f}", axis=1)
    latex["Macro-F1"] = latex.apply(lambda r: f"{r['macro_f1_mean']:.3f} $\\pm$ {r['macro_f1_std']:.3f}", axis=1)
    latex_out = latex[["modality_group", "strategy", "Degradation (%)", "Accuracy", "Macro-F1"]]
    latex_out = latex_out.rename(columns={"modality_group": "Modality group", "strategy": "Strategy"})
    (RESULT_DIR / "table_extended_degraded_latex.txt").write_text(latex_out.to_latex(index=False, escape=False), encoding="utf-8")

    _plot_extended_robustness(summary)

    print("\nExtended degraded-sensing summary:")
    print(summary)
    print("\nSaved:")
    print(" -", RESULT_DIR / "extended_degraded_results_by_seed.csv")
    print(" -", RESULT_DIR / "extended_degraded_results_summary.csv")
    print(" -", RESULT_DIR / "table_extended_degraded_latex.txt")
    print(" - figures in", FIGURE_DIR)


if __name__ == "__main__":
    main()
