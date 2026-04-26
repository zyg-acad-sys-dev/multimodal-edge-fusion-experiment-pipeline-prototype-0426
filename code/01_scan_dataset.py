import pandas as pd
from config import DATA_ROOT, INDEX_DIR
from utils_io import parse_sample_file

def scan_mat_modality(folder_name, suffix):
    folder = DATA_ROOT / folder_name
    out = {}
    if not folder.exists():
        print(f"[WARN] Missing folder: {folder}")
        return out
    for p in folder.glob(f"*_{suffix}.mat"):
        parsed = parse_sample_file(p)
        if parsed:
            out[parsed["sample_id"]] = {
                "action": parsed["action"], "subject": parsed["subject"], "trial": parsed["trial"],
                f"{suffix}_path": str(p),
            }
    return out

def scan_optional_rgb():
    out = {}
    for p in DATA_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".avi", ".mp4", ".mov"} and not p.name.startswith("._"):
            parsed = parse_sample_file(p)
            if parsed:
                out[parsed["sample_id"]] = {"rgb_path": str(p)}
    return out

def main():
    skel = scan_mat_modality("Skeleton", "skeleton")
    iner = scan_mat_modality("Inertial", "inertial")
    depth = scan_mat_modality("Depth", "depth")
    rgb = scan_optional_rgb()

    common = sorted(set(skel) & set(iner))
    rows = []
    for sid in common:
        rows.append({
            "sample_id": sid,
            "action": skel[sid]["action"],
            "subject": skel[sid]["subject"],
            "trial": skel[sid]["trial"],
            "skeleton_path": skel[sid]["skeleton_path"],
            "inertial_path": iner[sid]["inertial_path"],
            "depth_path": depth.get(sid, {}).get("depth_path", ""),
            "rgb_path": rgb.get(sid, {}).get("rgb_path", ""),
            "has_depth": sid in depth,
            "has_rgb": sid in rgb,
        })

    df = pd.DataFrame(rows).sort_values(["action", "subject", "trial"]).reset_index(drop=True)
    out = INDEX_DIR / "sample_index.csv"
    df.to_csv(out, index=False)

    print("Dataset scan complete.")
    print("Skeleton samples:", len(skel))
    print("Inertial samples:", len(iner))
    print("Depth samples:", len(depth))
    print("Optional RGB videos:", len(rgb))
    print("Common skeleton+inertial samples:", len(df))
    print("Saved:", out)
    print(df.head())

if __name__ == "__main__":
    main()
