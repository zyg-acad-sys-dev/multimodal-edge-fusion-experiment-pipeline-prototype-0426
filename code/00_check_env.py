import sys
import importlib
from config import DATA_ROOT, OUT_DIR

required = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]
optional = ["cv2"]

print("Python:", sys.version)
print("DATA_ROOT:", DATA_ROOT)
print("OUT_DIR:", OUT_DIR.resolve())

ok = True
print("\nRequired packages:")
for pkg in required:
    try:
        mod = importlib.import_module(pkg)
        print(f"  [OK] {pkg}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        ok = False
        print(f"  [MISSING] {pkg}: {e}")

print("\nOptional packages:")
for pkg in optional:
    try:
        mod = importlib.import_module(pkg)
        print(f"  [OK] {pkg}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"  [OPTIONAL MISSING] {pkg}: {e}")

if not DATA_ROOT.exists():
    print("\n[ERROR] DATA_ROOT does not exist. Edit config.py or set UTD_DATA_ROOT.")
    raise SystemExit(1)

if not ok:
    print("\nInstall missing packages with: pip install -r requirements.txt")
    raise SystemExit(1)

print("\nEnvironment check passed.")
