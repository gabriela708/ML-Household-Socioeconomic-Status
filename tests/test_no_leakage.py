"""Responsable: Gaby. Solo split.py (lo escribe) y evaluar_test.py (evaluación final, una vez) leen test_idx."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERMITIDOS = {"split.py", "evaluar_test.py"}

def test_nadie_lee_test_idx():
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if f.name in PERMITIDOS:
            continue
        assert "test_idx" not in f.read_text(encoding="utf-8"), f"{f} usa test_idx"
