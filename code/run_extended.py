"""Run optional extended-modality experiments after the core pipeline."""
import subprocess
import sys

for script in [
    "06_optional_extract_depth_rgb_features.py",
    "07_train_extended_modalities.py",
    "08_eval_extended_degraded_robustness.py",
]:
    print("\n" + "=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    subprocess.run([sys.executable, script], check=True)

print("\nAll extended steps completed.")
