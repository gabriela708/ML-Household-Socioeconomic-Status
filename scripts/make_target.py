"""Genera data/processed/y_hogar.parquet.
 
Uso:
    python scripts/make_target.py
 
"""
 
from ses.config import CFG, ROOT
from ses.data.target import build_target, COLUMNAS_CONTRATO
 
N_ESPERADO = 19690  # decisión #2
 
 
def main():
    y = build_target()
 
    # Controles mínimos antes de guardar
    assert list(y.columns) == COLUMNAS_CONTRATO, "Columnas distintas al contrato"
    assert y["hogar_id"].is_unique, "hogar_id duplicado"
    assert y["ses"].isin([0, 1, 2, 3]).all(), "Clase fuera de 0..3"
    if len(y) != N_ESPERADO:
        print(f"AVISO: {len(y)} hogares (se esperaban {N_ESPERADO})")
 
    out = ROOT / CFG["paths"]["processed"] / "y_hogar.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    y.to_parquet(out, index=False)
 
    dist = (y["ses"].map(CFG["clases"]).value_counts()
            .reindex(CFG["clases"].values()).rename("hogares").to_frame())
    dist["pct"] = (100 * dist["hogares"] / len(y)).round(1)
    print(f"Guardado {out.relative_to(ROOT)}  ({len(y)} hogares)")
    print(dist.to_string())
 
 
if __name__ == "__main__":
    main()
 
