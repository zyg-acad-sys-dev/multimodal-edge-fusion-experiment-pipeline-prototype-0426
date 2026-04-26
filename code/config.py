from pathlib import Path
import os

DATA_ROOT = Path(
    os.environ.get("DATA_ROOT")
    or os.environ.get("UTD_DATA_ROOT")
    or "./data"
)

OUT_DIR = Path("outputs")
INDEX_DIR = OUT_DIR / "index"
FEATURE_DIR = OUT_DIR / "features"
RESULT_DIR = OUT_DIR / "results"
FIGURE_DIR = OUT_DIR / "figures"

for d in [OUT_DIR, INDEX_DIR, FEATURE_DIR, RESULT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_SUBJECTS = {1, 2, 3, 4}
TEST_SUBJECTS = {5, 6, 7, 8}
RANDOM_SEEDS = [11, 22, 33, 44, 55]
DEGRADATION_RATES = [0.0, 0.1, 0.3, 0.5]
MODEL_TYPE = "rf"  # "rf" or "lr"
ENABLE_DEPTH_FEATURES = False
ENABLE_RGB_FEATURES = False