"""Partición 80/20 estratificada sin separar viviendas (DIRECTORIO) y folds de VC. Guarda data/processed/split_indices.json.

Responsable: Juanita. Día 2.

Entrada:  data/processed/X_hogar.parquet  (Juanita)
          data/processed/y_hogar.parquet  (Silva)
Salida:   data/processed/split_indices.json, con la forma del contrato de datos:
          {"seed": 42, "train_idx": [hogar_id...], "test_idx": [hogar_id...],
           "folds": [[hogar_id de validación del fold 1], ..., [fold 5]]}

Reglas:
- Estratificado por `ses`: cada parte conserva la proporción de las 4 clases.
- Agrupado por `DIRECTORIO`: los hogares de una misma vivienda quedan siempre del mismo lado.
- El 20 % de prueba queda sellado hasta la entrega final (decisión 7).
- Los folds se arman solo con el 80 % de entrenamiento.

Uso:      python src/ses/features/split.py
"""

import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ses.config import CFG, ROOT, SEED

PROCESSED = ROOT / CFG["paths"]["processed"]
SALIDA = PROCESSED / "split_indices.json"


def cargar_datos():
    """Une X e y por hogar_id. Solo quedan hogares con predictores y con target."""
    X = pd.read_parquet(PROCESSED / "X_hogar.parquet", columns=["hogar_id"])
    y = pd.read_parquet(PROCESSED / "y_hogar.parquet", columns=["hogar_id", "DIRECTORIO", "ses"])
    datos = X.merge(y, on="hogar_id", how="inner", validate="one_to_one")
    if len(datos) != len(X):
        raise ValueError(f"{len(X) - len(datos)} hogares de X no tienen target")
    return datos.sort_values("hogar_id").reset_index(drop=True)


def separar_prueba(datos):
    """80/20: se toma 1 de 5 grupos estratificados como prueba (20 %)."""
    n_partes = round(1 / CFG["split"]["test_size"])          # 0.20 -> 5 partes
    sgkf = StratifiedGroupKFold(n_splits=n_partes, shuffle=True, random_state=SEED)
    pos_train, pos_test = next(sgkf.split(datos, datos["ses"], groups=datos[CFG["split"]["group"]]))
    return datos.iloc[pos_train].reset_index(drop=True), datos.iloc[pos_test].reset_index(drop=True)


def armar_folds(train):
    """5 folds de validación cruzada dentro del entrenamiento, estratificados y agrupados."""
    sgkf = StratifiedGroupKFold(n_splits=CFG["split"]["n_folds"], shuffle=True, random_state=SEED)
    return [train["hogar_id"].iloc[pos_val].tolist()
            for _, pos_val in sgkf.split(train, train["ses"], groups=train[CFG["split"]["group"]])]


def verificar(datos, train, test, folds):
    """Controles antes de guardar: nada se repite, nada se pierde y ninguna vivienda queda partida."""
    ids_train, ids_test = set(train["hogar_id"]), set(test["hogar_id"])
    assert not ids_train & ids_test, "hay hogares en train y en test"
    assert len(ids_train) + len(ids_test) == len(datos), "se perdieron hogares"
    assert not set(train["DIRECTORIO"]) & set(test["DIRECTORIO"]), "una vivienda quedó partida"

    ids_folds = [h for f in folds for h in f]
    assert len(ids_folds) == len(set(ids_folds)), "un hogar está en dos folds"
    assert set(ids_folds) == ids_train, "los folds no cubren exactamente el entrenamiento"


def resumen(datos, train, test, folds):
    """Tabla con el % de cada clase en cada parte, para comprobar la estratificación."""
    partes = {"Población": datos, "Entrenamiento": train, "Prueba": test}
    por_id = train.set_index("hogar_id")["ses"]
    for i, f in enumerate(folds, start=1):
        partes[f"Fold {i}"] = por_id.loc[f].to_frame()
    tabla = pd.DataFrame({
        nombre: (100 * df["ses"].value_counts(normalize=True)).round(1)
        for nombre, df in partes.items()
    }).T.reindex(columns=list(CFG["clases"])).rename(columns=CFG["clases"])
    tabla.insert(0, "hogares", [len(df) for df in partes.values()])
    return tabla


def main():
    datos = cargar_datos()
    train, test = separar_prueba(datos)
    folds = armar_folds(train)
    verificar(datos, train, test, folds)

    split = {
        "seed": SEED,
        "train_idx": train["hogar_id"].tolist(),
        "test_idx": test["hogar_id"].tolist(),
        "folds": folds,
    }
    SALIDA.write_text(json.dumps(split), encoding="utf-8")

    print(f"Guardado {SALIDA.relative_to(ROOT)}")
    print(resumen(datos, train, test, folds).to_string())


if __name__ == "__main__":
    main()