import subprocess
import sys

steps = [
    "00_check_env.py",
    "01_scan_dataset.py",
    "02_extract_features.py",
    "03_train_clean_baselines.py",
    "04_eval_degraded_sensing.py",
    "05_export_tables_figures.py",
]

for step in steps:
    print("\n" + "=" * 80)
    print("Running", step)
    print("=" * 80)
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"[STOP] {step} failed with code {result.returncode}")
        raise SystemExit(result.returncode)

print("\nAll core steps completed.")
