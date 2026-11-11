"""PPOM: validación cruzada (A+B y A+B+C) y comparación con el POM como en el paper.


No vuelve a correr los modelos que ya están en reports/tables/: evalúa solo el PPOM
con el mismo 80 % de entrenamiento, los mismos 5 folds (split_indices.json), el mismo
pipeline y las mismas métricas, y agrega sus filas a las tablas existentes. Eso es
válido (no hay fuga ni comparación injusta) mientras X_hogar, y_hogar y
split_indices.json no hayan cambiado desde que se corrió run_cv.py. 
El conjunto de prueba no se toca.

Uso:
    python scripts/run_ppom.py
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from ses.config import CFG, ROOT
from ses.evaluation.metrics import METRICA_PRINCIPAL, SCORERS, a_signo_normal
from ses.models.candidates import construir_busqueda
from ses.models.ordinal import preparar_X_inferencia, tablas_ppom
from ses.pipeline import cargar_entrenamiento

warnings.filterwarnings("ignore", message="Found unknown categories")

PROCESSED = ROOT / CFG["paths"]["processed"]
TABLAS = ROOT / "reports" / "tables"
NOMBRE = "ppom"
BLOQUES = ["B", "C"]
N_FOLDS = 5


def evaluar_bloque(bloque, X, y, cv):
    """GridSearchCV + puntajes por fold del mejor + predicciones y diagnóstico por fold."""
    busqueda = construir_busqueda(NOMBRE, cv, bloque=bloque).fit(X, y)
    r, i = busqueda.cv_results_, busqueda.best_index_

    filas = []
    for m in SCORERS:
        for k in range(N_FOLDS):
            filas.append({"modelo": NOMBRE, "bloque": bloque, "fold": k + 1, "metrica": m,
                          "valor": a_signo_normal(m, r[f"split{k}_test_{m}"][i])})

    # Mismo mejor modelo, reentrenado en cada fold para guardar predicciones y diagnóstico
    res = cross_validate(busqueda.best_estimator_, X, y, cv=cv, return_estimator=True)
    pred = np.empty(len(y), dtype=int)
    diagnostico = []
    for k, ((_, val), est) in enumerate(zip(cv, res["estimator"]), start=1):
        Xval = est.named_steps["prep"].transform(X.iloc[val])
        pred[val] = est.named_steps["modelo"].predict(Xval)
        modelo = est.named_steps["modelo"]
        diagnostico.append({"bloque": bloque, "fold": k,
                            "n_variables": modelo.n_features_in_,
                            "n_relajadas": modelo.n_relajadas_,
                            "convergio": modelo.convergio_,
                            "tasa_cruces_validacion": modelo.tasa_cruces(Xval)})

    detalle = pd.DataFrame(r["params"]).astype(str).add_prefix("param_")
    for m in SCORERS:
        detalle[f"{m}_media"] = a_signo_normal(m, 1) * r[f"mean_test_{m}"]
        detalle[f"{m}_desv"] = r[f"std_test_{m}"]
    detalle.insert(0, "modelo", NOMBRE)

    medias = np.sort(r[f"mean_test_{METRICA_PRINCIPAL}"])[::-1]
    info = {"modelo": NOMBRE, "hiperparametros": str(busqueda.best_params_),
            "diferencia_con_segundo": round(medias[0] - medias[1], 4)}
    return pd.DataFrame(filas), pred, pd.DataFrame(diagnostico), detalle, info


def reemplazar(ruta, nuevas, clave="modelo", dtype=None):
    """Quita las filas viejas del PPOM (si las hay) y agrega las nuevas.

    dtype=str para grilla_detalle.csv: así pandas no convierte k = 5 en 5.0.
    """
    vieja = pd.read_csv(ruta, dtype=dtype)
    vieja = vieja[vieja[clave] != NOMBRE]
    tabla = pd.concat([vieja, nuevas], ignore_index=True)
    tabla.to_csv(ruta, index=False)
    return tabla


def main():
    X, y, cv = cargar_entrenamiento()
    print(f"Entrenamiento: {X.shape[0]} hogares, {len(cv)} folds\n")

    resultados = {}
    for bloque in BLOQUES:
        t = time.time()
        resultados[bloque] = evaluar_bloque(bloque, X, y, cv)
        filas, _, diag, _, info = resultados[bloque]
        f1 = filas.loc[filas["metrica"] == METRICA_PRINCIPAL, "valor"].mean()
        print(f"PPOM bloque {bloque}: macro-F1 = {f1:.3f}   {info['hiperparametros']}   "
              f"relajadas por fold = {diag['n_relajadas'].tolist()}   ({time.time() - t:.0f} s)")

    # Tablas de la comparación de modelos (solo bloque C, igual que run_cv.py)
    filas_c, pred_c, _, detalle_c, info_c = resultados["C"]
    cv_tabla = None
    for ruta in [PROCESSED / "cv_resultados.csv", TABLAS / "cv_resultados.csv"]:
        cv_tabla = reemplazar(ruta, filas_c)
    resumen = (cv_tabla.groupby(["modelo", "metrica"], sort=False)["valor"]
               .agg(media="mean", desv="std").round(3).reset_index())
    resumen.to_csv(TABLAS / "cv_resumen.csv", index=False)

    f1_c = filas_c.loc[filas_c["metrica"] == METRICA_PRINCIPAL, "valor"]
    info_c.update({"f1_macro_media": round(f1_c.mean(), 3), "f1_macro_desv": round(f1_c.std(), 3)})
    reemplazar(TABLAS / "mejores_hiperparametros.csv", pd.DataFrame([info_c]))
    reemplazar(TABLAS / "grilla_detalle.csv", detalle_c.round(4), dtype=str)

    pred = pd.read_csv(TABLAS / "predicciones_cv.csv")
    assert (pred["hogar_id"].astype(str).to_numpy() == X.index.astype(str).to_numpy()).all(), \
        "predicciones_cv.csv no tiene los mismos hogares en el mismo orden: corre run_cv.py"
    pred[NOMBRE] = pred_c
    pred.to_csv(TABLAS / "predicciones_cv.csv", index=False)

    # Tablas de la ablación (bloques B y C)
    filas_bc = pd.concat([resultados[b][0] for b in BLOQUES], ignore_index=True)
    abl = reemplazar(TABLAS / "ablacion.csv", filas_bc)
    (abl.groupby(["modelo", "bloque", "metrica"])["valor"]
        .agg(media="mean", desv="std").round(3).reset_index()
        .to_csv(TABLAS / "ablacion_resumen.csv", index=False))

    # Diagnóstico por fold
    diag = pd.concat([resultados[b][2] for b in BLOQUES], ignore_index=True)
    diag.to_csv(TABLAS / "ppom_diagnostico_folds.csv", index=False)

    # Inferencia como en el paper (razón de verosimilitud y AIC)
    print("\nInferencia POM vs. PPOM con el 80 % completo (puede tardar unos minutos)...")
    tabla5, tabla6 = tablas_ppom(preparar_X_inferencia(X), y.to_numpy())
    tabla5.round(3).to_csv(TABLAS / "tabla5_ppom_variables.csv")
    tabla6.round(4).to_csv(TABLAS / "tabla6_pom_vs_ppom.csv")


    ancho = resumen.pivot(index="modelo", columns="metrica", values="media")[list(SCORERS)]
    print("\nComparación de modelos, bloque C (media en 5 folds; mae y errores lejanos: menor es mejor)")
    print(ancho.sort_values(METRICA_PRINCIPAL, ascending=False).to_string())
    print("\nDiagnóstico por fold")
    print(diag.to_string(index=False))
    print("\nPOM vs. PPOM (paper)")
    print(tabla6.T.to_string())
    print(f"\nVariables relajadas en la inferencia: {len(tabla5)}")


if __name__ == "__main__":
    main()