"""Validación cruzada de todos los modelos (bloque C) y tablas para el informe.

1. Línea base (clase mayoritaria) y POM (modelo principal, de Silva, sin ajustar:
   alpha = 0 como en el paper, decisión 34).
2. Cada candidato de ses.models.candidates con GridSearchCV: elige el mejor
   hiperparámetro por macro-F1 y guarda los 5 puntajes de ese mejor modelo.
3. Predicciones fuera de partición (cross_val_predict) de cada modelo, para las
   matrices de confusión y el análisis de errores (plots.py).

Todo usa el 80 % de entrenamiento y los mismos 5 folds (cargar_entrenamiento()).
El conjunto de prueba no se toca.

Salidas:
    data/processed/cv_resultados.csv          contrato de datos: modelo, bloque, fold, metrica, valor
    reports/tables/cv_resultados.csv          la misma tabla (copia versionada en git)
    reports/tables/cv_resumen.csv             media y desviación por modelo y métrica
    reports/tables/mejores_hiperparametros.csv
    reports/tables/grilla_detalle.csv         todas las combinaciones probadas
    reports/tables/predicciones_cv.csv        hogar_id, ses real y predicción de cada modelo

Uso:
    python scripts/run_cv.py
    python -m ses.evaluation.plots
"""

import time
import warnings
import sklearn

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_validate

from ses.config import CFG, ROOT
from ses.evaluation.metrics import METRICA_PRINCIPAL, SCORERS, a_signo_normal
from ses.models.baseline import construir_baseline
from ses.models.candidates import BLOQUE_COMPARACION, CANDIDATOS, construir_busqueda
from ses.models.ordinal import POM
from ses.pipeline import cargar_entrenamiento, construir_pipeline

warnings.filterwarnings("ignore", message="Found unknown categories")

PROCESSED = ROOT / CFG["paths"]["processed"]
TABLAS = ROOT / "reports" / "tables"
N_FOLDS = 5


def filas_contrato(modelo, puntajes):
    """puntajes: {metrica: [valor_fold1, ...]} con el signo de scikit-learn."""
    filas = []
    for metrica, valores in puntajes.items():
        for fold, valor in enumerate(valores, start=1):
            filas.append({"modelo": modelo, "bloque": BLOQUE_COMPARACION, "fold": fold,
                          "metrica": metrica, "valor": a_signo_normal(metrica, valor)})
    return filas


def evaluar_fijo(nombre, modelo, X, y, cv):
    """Modelos sin grilla (línea base y POM): cross_validate + predicciones."""
    pipe = construir_pipeline(modelo, bloque=BLOQUE_COMPARACION)
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS)
    puntajes = {m: res[f"test_{m}"] for m in SCORERS}
    pred = cross_val_predict(pipe, X, y, cv=cv)
    info = {"modelo": nombre, "hiperparametros": "fijos", "diferencia_con_segundo": np.nan}
    return filas_contrato(nombre, puntajes), pred, info, None


def evaluar_candidato(nombre, X, y, cv):
    """Candidatos con grilla: GridSearchCV + puntajes por fold del mejor + predicciones."""
    busqueda = construir_busqueda(nombre, cv).fit(X, y)
    r, i = busqueda.cv_results_, busqueda.best_index_
    puntajes = {m: [r[f"split{k}_test_{m}"][i] for k in range(N_FOLDS)] for m in SCORERS}

    medias = np.sort(r[f"mean_test_{METRICA_PRINCIPAL}"])[::-1]
    diferencia = medias[0] - medias[1] if len(medias) > 1 else np.nan

    pred = cross_val_predict(busqueda.best_estimator_, X, y, cv=cv)

    detalle = pd.DataFrame(r["params"]).astype(str).add_prefix("param_")
    for m in SCORERS:
        signo = a_signo_normal(m, 1)
        detalle[f"{m}_media"] = signo * r[f"mean_test_{m}"]
        detalle[f"{m}_desv"] = r[f"std_test_{m}"]
    detalle.insert(0, "modelo", nombre)

    info = {"modelo": nombre, "hiperparametros": str(busqueda.best_params_) if busqueda.best_params_ else "sin grilla",
            "diferencia_con_segundo": diferencia}
    return filas_contrato(nombre, puntajes), pred, info, detalle


def main():
    X, y, cv = cargar_entrenamiento()
    print(f"Entrenamiento: {X.shape[0]} hogares, {X.shape[1]} variables, {len(cv)} folds\n")

    filas, infos, detalles = [], [], []
    predicciones = pd.DataFrame({"hogar_id": X.index, "ses_real": y.to_numpy()})

    trabajos = [
        ("baseline_mayoritaria", lambda: evaluar_fijo("baseline_mayoritaria", construir_baseline(), X, y, cv)),
        ("pom", lambda: evaluar_fijo("pom", POM(class_weight="balanced"), X, y, cv)),
    ] + [(n, lambda n=n: evaluar_candidato(n, X, y, cv)) for n in CANDIDATOS]

    for nombre, trabajo in trabajos:
        t = time.time()
        f, pred, info, detalle = trabajo()
        filas += f
        infos.append(info)
        predicciones[nombre] = pred
        if detalle is not None:
            detalles.append(detalle)
        f1 = np.mean([x["valor"] for x in f if x["metrica"] == METRICA_PRINCIPAL])
        print(f"{nombre:22s} macro-F1 = {f1:.3f}   {info['hiperparametros']}   ({time.time() - t:.0f} s)")

    # Tablas
    TABLAS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(PROCESSED / "cv_resultados.csv", index=False)
    tabla.to_csv(TABLAS / "cv_resultados.csv", index=False)

    resumen = (tabla.groupby(["modelo", "metrica"], sort=False)["valor"]
               .agg(media="mean", desv="std").round(3).reset_index())
    resumen.to_csv(TABLAS / "cv_resumen.csv", index=False)

    mejores = pd.DataFrame(infos)
    f1 = resumen[resumen["metrica"] == METRICA_PRINCIPAL].set_index("modelo")
    mejores["f1_macro_media"] = mejores["modelo"].map(f1["media"])
    mejores["f1_macro_desv"] = mejores["modelo"].map(f1["desv"])
    mejores["diferencia_con_segundo"] = mejores["diferencia_con_segundo"].round(4)
    mejores.to_csv(TABLAS / "mejores_hiperparametros.csv", index=False)

    pd.concat(detalles, ignore_index=True).round(4).to_csv(TABLAS / "grilla_detalle.csv", index=False)
    predicciones.to_csv(TABLAS / "predicciones_cv.csv", index=False)

    ancho = resumen.pivot(index="modelo", columns="metrica", values="media")[list(SCORERS)]
    ancho = ancho.sort_values(METRICA_PRINCIPAL, ascending=False)
    print("\nMedia en 5 folds (bloque C; mae y errores lejanos: menor es mejor)")
    print(ancho.to_string())
    print("\nGuardado en reports/tables/ y data/processed/cv_resultados.csv")


if __name__ == "__main__":
    main()