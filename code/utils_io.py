import re
from pathlib import Path
from scipy.io import loadmat

def parse_sample_file(path: Path):
    name = path.name
    m = re.match(r"a(\d+)_s(\d+)_t(\d+)_(skeleton|inertial|depth)\.mat$", name, re.I)
    if m:
        a, s, t, mod = m.groups()
        return {
            "sample_id": f"a{int(a)}_s{int(s)}_t{int(t)}",
            "action": int(a), "subject": int(s), "trial": int(t),
            "modality": mod.lower(),
        }

    # Optional RGB/video file names are allowed to be less strict.
    m = re.match(r"a(\d+)_s(\d+)_t(\d+).*\.(avi|mp4|mov)$", name, re.I)
    if m:
        a, s, t, _ = m.groups()
        return {
            "sample_id": f"a{int(a)}_s{int(s)}_t{int(t)}",
            "action": int(a), "subject": int(s), "trial": int(t),
            "modality": "rgb",
        }
    return None

def load_mat_var(path: Path, expected_key: str):
    data = loadmat(path)
    if expected_key in data:
        return data[expected_key]
    keys = [k for k in data.keys() if not k.startswith("__")]
    if len(keys) == 1:
        return data[keys[0]]
    raise KeyError(f"Cannot find {expected_key} in {path}. Available keys: {keys}")
