"""Carga config/config.yaml."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=ROOT / "config" / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()
SEED = CFG["seed"]
