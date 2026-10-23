"""Responsable: Gaby. Ningún script de la 2.a entrega puede leer test_idx."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERMITIDOS = {"split.py"}  # único archivo que escribe test_idx

def test_nadie_lee_test_idx():
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if f.name in PERMITIDOS:
            continue
        assert "test_idx" not in f.read_text(encoding="utf-8"), f"{f} usa test_idx"
