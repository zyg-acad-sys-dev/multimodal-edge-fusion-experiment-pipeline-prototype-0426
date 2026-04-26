import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FEATURE_DIR, RESULT_DIR, FIGURE_DIR

def fmt(mean, std):
    if pd.isna(std):
        std = 0.0
    return f"{mean:.3f} ± {std:.3f}"

def export_latex_tables():
    clean_path = RESULT_DIR / "clean_results_summary.csv"
    deg_path = RESULT_DIR / "degraded_results_summary.csv"

    if clean_path.exists():
        clean = pd.read_csv(clean_path)
        clean["Accuracy"] = clean.apply(lambda r: fmt(r["accuracy_mean"], r["accuracy_std"]), axis=1)
        clean["Macro-F1"] = clean.apply(lambda r: fmt(r["macro_f1_mean"], r["macro_f1_std"]), axis=1)
        out = clean[["model", "Accuracy", "Macro-F1"]].rename(columns={"model": "Model"})
        (RESULT_DIR / "table_clean_latex.txt").write_text(out.to_latex(index=False, escape=False), encoding="utf-8")

    if deg_path.exists():
        deg = pd.read_csv(deg_path)
        deg["Degradation (%)"] = (deg["degradation_rate"] * 100).astype(int)
        deg["Macro-F1"] = deg.apply(lambda r: fmt(r["macro_f1_mean"], r["macro_f1_std"]), axis=1)
        out = deg[["Degradation (%)", "model", "Macro-F1"]].rename(columns={"model": "Model"})
        (RESULT_DIR / "table_degraded_latex.txt").write_text(out.to_latex(index=False, escape=False), encoding="utf-8")

def plot_robustness():
    deg_path = RESULT_DIR / "degraded_results_summary.csv"
    if not deg_path.exists():
        print("[WARN] Missing degraded_results_summary.csv.")
        return

    df = pd.read_csv(deg_path)
    plt.figure(figsize=(6.5, 4.2))

    order = ["Skeleton-only", "Inertial-only", "Early fusion", "Reliability-aware fusion"]
    for model in order:
        sub = df[df["model"] == model].sort_values("degradation_rate")
        if sub.empty:
            continue
        x = sub["degradation_rate"].values * 100
        y = sub["macro_f1_mean"].values
        s = sub["macro_f1_std"].fillna(0).values
        plt.plot(x, y, marker="o", label=model)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("Simulated sensing degradation rate (%)")
    plt.ylabel("Macro-F1")
    plt.title("Robustness under degraded sensing")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "robustness_curve.pdf")
    plt.savefig(FIGURE_DIR / "robustness_curve.png", dpi=300)
    plt.close()

def export_payload_estimate():
    Xs = np.load(FEATURE_DIR / "X_skeleton.npy")
    Xi = np.load(FEATURE_DIR / "X_inertial.npy")

    raw_payload = int((Xs.shape[1] + Xi.shape[1]) * 4)
    event_payload = 32

    df = pd.DataFrame([
        {"mode": "Raw feature upload", "uploaded_content": "skeleton features + inertial features",
         "estimated_bytes_per_sample": raw_payload},
        {"mode": "Edge event summary", "uploaded_content": "event id + confidence + timestamp + reliability scores",
         "estimated_bytes_per_sample": event_payload},
        {"mode": "Estimated reduction ratio", "uploaded_content": "raw payload / event payload",
         "estimated_bytes_per_sample": round(raw_payload / event_payload, 2)},
    ])
    df.to_csv(RESULT_DIR / "payload_estimate.csv", index=False)
    (RESULT_DIR / "table_payload_latex.txt").write_text(df.to_latex(index=False, escape=False), encoding="utf-8")

def main():
    export_latex_tables()
    plot_robustness()
    export_payload_estimate()
    print("Export complete.")
    print("Figures:", FIGURE_DIR.resolve())
    print("Results:", RESULT_DIR.resolve())

if __name__ == "__main__":
    main()
