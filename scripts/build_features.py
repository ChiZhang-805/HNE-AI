from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_features import build_features

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    build_features(ROOT/"data/development", ROOT/"data/features")
