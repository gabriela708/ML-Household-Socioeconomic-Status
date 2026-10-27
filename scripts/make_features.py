"""Genera data/processed/X_hogar.parquet y split_indices.json.

Responsable: Juanita. Día 2.

Uso (después de scripts/make_target.py):
    python scripts/make_features.py
"""

from ses.features.build_features import construir_X, guardar
from ses.features.split import main as hacer_split


def main():
    # 1. Predictores del hogar: lo mismo que hace el final de build_features.py
    X_final, excluidos, embudo = construir_X()
    guardar(X_final, excluidos, embudo)

    for paso, n in embudo.items():
        print(f"{paso}: {n:,}")
    print("Excluidos por vacíos:", len(excluidos))
    print("Sin fuga ✔ — guardado data/processed/X_hogar.parquet", X_final.shape)

    # 2. Partición 80/20 y folds: lee X_hogar y y_hogar, por eso va después
    hacer_split()


if __name__ == "__main__":
    main()