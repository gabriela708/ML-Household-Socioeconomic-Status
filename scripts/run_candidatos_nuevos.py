"""Agrega candidatos nuevos a la comparación sin volver a correr los que ya están.


Usa exactamente la misma función de run_cv.py (evaluar_candidato): mismo 80 %, mismos
5 folds, mismo pipeline, mismas métricas y bloque C. Solo agrega (o reemplaza) las
filas de los modelos pedidos en las tablas de reports/tables/. Válido mientras
X_hogar, y_hogar y split_indices.json no cambien.

Uso:
    python scripts/run_candidatos_nuevos.py lda qda
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

from ses.evaluation.metrics import METRICA_PRINCIPAL, SCORERS
from ses.models.candidates import CANDIDATOS
from ses.pipeline import cargar_entrenamiento
from run_cv import PROCESSED, TABLAS, evaluar_candidato

warnings.filterwarnings("ignore", message="Found unknown categories")
warnings.filterwarnings("ignore", message="Variables are collinear")


def reemplazar(ruta, nuevas, nombres, dtype=None):
    """dtype=str para grilla_detalle.csv: así pandas no convierte k = 5 en 5.0."""
    vieja = pd.read_csv(ruta, dtype=dtype)
    tabla = pd.concat([vieja[~vieja["modelo"].isin(nombres)], nuevas], ignore_index=True)
    tabla.to_csv(ruta, index=False)
    return tabla


def main(nombres):
    faltan = [n for n in nombres if n not in CANDIDATOS]
    if faltan:
        sys.exit(f"No están en CANDIDATOS: {faltan}. Opciones: {list(CANDIDATOS)}")

    X, y, cv = cargar_entrenamiento()
    pred = pd.read_csv(TABLAS / "predicciones_cv.csv")
    assert (pred["hogar_id"].astype(str).to_numpy() == X.index.astype(str).to_numpy()).all(), \
        "predicciones_cv.csv no tiene los mismos hogares en el mismo orden: corre run_cv.py"

    filas, infos, detalles = [], [], []
    for nombre in nombres:
        t = time.time()
        f, p, info, detalle = evaluar_candidato(nombre, X, y, cv)
        filas += f
        pred[nombre] = p
        detalles.append(detalle)
        f1 = [x["valor"] for x in f if x["metrica"] == METRICA_PRINCIPAL]
        info.update({"f1_macro_media": round(np.mean(f1), 3), "f1_macro_desv": round(np.std(f1, ddof=1), 3)})
        infos.append(info)
        print(f"{nombre:8s} macro-F1 = {np.mean(f1):.3f}   {info['hiperparametros']}   ({time.time() - t:.0f} s)")

    nuevas = pd.DataFrame(filas)
    for ruta in [PROCESSED / "cv_resultados.csv", TABLAS / "cv_resultados.csv"]:
        tabla = reemplazar(ruta, nuevas, nombres)
    resumen = (tabla.groupby(["modelo", "metrica"], sort=False)["valor"]
               .agg(media="mean", desv="std").round(3).reset_index())
    resumen.to_csv(TABLAS / "cv_resumen.csv", index=False)
    reemplazar(TABLAS / "mejores_hiperparametros.csv", pd.DataFrame(infos), nombres)
    reemplazar(TABLAS / "grilla_detalle.csv", pd.concat(detalles).round(4), nombres, dtype=str)
    pred.to_csv(TABLAS / "predicciones_cv.csv", index=False)

    ancho = resumen.pivot(index="modelo", columns="metrica", values="media")[list(SCORERS)]
    print("\nMedia en 5 folds (bloque C; mae y errores lejanos: menor es mejor)")
    print(ancho.sort_values(METRICA_PRINCIPAL, ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["lda", "qda"])