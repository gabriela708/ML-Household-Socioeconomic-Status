"""Evaluación final en el conjunto de prueba (20 %). Se corre UNA sola vez.

Modelo final: PPOM . Escenario final: A+B+C y A+B como referencia .

Pasos:
1. Entrena el PPOM UNA vez por escenario con todo el 80 %, con el alpha_gamma ya elegido
   en la validación cruzada (run_ppom.py). No se vuelve a elegir nada: en la CV hubo 5
   versiones del modelo (una por fold) y ninguna usó el 80 % completo; este es el modelo
   final que se evalúa y se guarda para la app. El test no participa en ninguna decisión.
2. Abre el 20 % de prueba y evalúa: el PPOM de cada escenario y la línea base.
3. Intervalos de confianza del 95 % por bootstrap (1.000 remuestreos de hogares de prueba).
4. Guarda los modelos entrenados para la app de Streamlit.

Sello: al terminar escribe reports/tables/test_sello.json. Si el sello existe, el script
se niega a correr de nuevo. Solo se puede repetir con --forzar.

Salidas:
    reports/tables/test_metricas.csv       bloque, modelo, métrica, valor en prueba, IC 95 %, media en CV
    reports/tables/test_confusion.csv      matriz de confusión (conteos) de cada escenario
    reports/tables/test_errores.csv        errores graves en las dos direcciones
    reports/tables/test_predicciones.csv   hogar_id, ses real, predicción y probabilidades
    reports/tables/test_sello.json         fecha, hiperparámetros y tamaños
    reports/figures/test_confusion.png     matrices de confusión (% por fila)
    models/ppom_AB.joblib, models/ppom_ABC.joblib   Pipeline completo (prep + PPOM)

Uso:
    python scripts/evaluar_test.py
"""

import json
import sys
import warnings
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from ses.config import CFG, ROOT, SEED
from ses.evaluation.metrics import CLASES, NOMBRES_CLASES, resumen_metricas
from ses.evaluation.plots import AZULES, TINTA, matriz_porcentaje
from ses.models.ordinal import PPOM
from ses.pipeline import cargar_entrenamiento, construir_pipeline

warnings.filterwarnings("ignore", message="Found unknown categories")

PROCESSED = ROOT / CFG["paths"]["processed"]
TABLAS = ROOT / "reports" / "tables"
FIGURAS = ROOT / "reports" / "figures"
MODELOS = ROOT / "models"
SELLO = TABLAS / "test_sello.json"

ESCENARIOS = {"C": "A+B+C (final)", "B": "A+B (referencia)"}

ALPHA_GAMMA = {"C": 1, "B": 10}
ARCHIVO_MODELO = {"C": "ppom_ABC.joblib", "B": "ppom_AB.joblib"}
N_BOOTSTRAP = 1000


def cargar_prueba():
    """X, y del 20 % de prueba. Único lugar del proyecto (además de split.py) que lo lee."""
    split = json.loads((PROCESSED / "split_indices.json").read_text(encoding="utf-8"))
    ids = split["test_idx"]
    X = pd.read_parquet(PROCESSED / "X_hogar.parquet").set_index("hogar_id").loc[ids]
    y = pd.read_parquet(PROCESSED / "y_hogar.parquet").set_index("hogar_id")["ses"].loc[ids]
    return X, y


def bootstrap_ic(y_real, y_pred, rng):
    """IC 95 % percentil de cada métrica, remuestreando hogares de prueba."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    n = len(y_real)
    muestras = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, n)
        muestras.append(resumen_metricas(y_real[i], y_pred[i]))
    muestras = pd.DataFrame(muestras)
    return muestras.quantile(0.025), muestras.quantile(0.975)


def errores_graves(y_real, y_pred):
    """Los dos errores que importan para focalización."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    pobres, media_alta = y_real == 0, y_real >= 2
    return {
        "pobres": int(pobres.sum()),
        "pobre_a_media_alta_n": int((pobres & (y_pred >= 2)).sum()),
        "pobre_a_media_alta_pct": round(100 * (pobres & (y_pred >= 2)).sum() / pobres.sum(), 2),
        "media_alta": int(media_alta.sum()),
        "media_alta_a_pobre_n": int((media_alta & (y_pred == 0)).sum()),
        "media_alta_a_pobre_pct": round(100 * (media_alta & (y_pred == 0)).sum() / media_alta.sum(), 2),
    }


def media_cv(bloque):
    """Media en validación cruzada del PPOM en ese bloque (de run_ppom.py), para comparar."""
    ruta = TABLAS / "ablacion_resumen.csv"
    if not ruta.exists():
        return {}
    r = pd.read_csv(ruta)
    r = r[(r["modelo"] == "ppom") & (r["bloque"] == bloque)]
    return r.set_index("metrica")["media"].to_dict()


def figura_confusion(y_real, predicciones, ruta):
    nombres = [NOMBRES_CLASES[c] for c in CLASES]
    fig, ejes = plt.subplots(1, len(predicciones), figsize=(4.4 * len(predicciones), 3.9))
    for ax, (titulo, y_pred) in zip(np.atleast_1d(ejes), predicciones.items()):
        cm = matriz_porcentaje(y_real, y_pred)
        ax.imshow(cm, cmap=AZULES, vmin=0, vmax=100)
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                ax.text(j, i, f"{cm[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 55 else TINTA)
        ax.set_xticks(range(len(CLASES)), nombres, fontsize=8)
        ax.set_yticks(range(len(CLASES)), nombres, fontsize=8)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(titulo, fontsize=10)
    fig.suptitle("PPOM en el conjunto de prueba (% de cada clase real)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(forzar=False):
    if SELLO.exists() and not forzar:
        sys.exit(f"El conjunto de prueba ya se usó ({SELLO.name}). No se vuelve a correr.\n"
                 "Si de verdad hay que repetirlo, usa --forzar.")


    archivo = TABLAS / "mejores_hiperparametros.csv"
    if archivo.exists():
        h = pd.read_csv(archivo).set_index("modelo")["hiperparametros"]
        if "ppom" in h and str(ALPHA_GAMMA["C"]) not in h["ppom"]:
            sys.exit(f"ALPHA_GAMMA['C'] = {ALPHA_GAMMA['C']} no coincide con {archivo.name}: {h['ppom']}")

    X_tr, y_tr, _ = cargar_entrenamiento()
    finales, hiper = {}, dict(ALPHA_GAMMA)
    for bloque in ESCENARIOS:
        modelo = PPOM(alpha=0.0, alpha_gamma=ALPHA_GAMMA[bloque], class_weight="balanced")
        finales[bloque] = construir_pipeline(modelo, bloque=bloque).fit(X_tr, y_tr)   # una vez, 80 %
        modelo = finales[bloque].named_steps["modelo"]
        print(f"PPOM {ESCENARIOS[bloque]:18s} alpha_gamma = {hiper[bloque]}   "
              f"variables relajadas = {modelo.n_relajadas_}   convergió = {modelo.convergio_}")
    base = construir_pipeline(DummyClassifier(strategy="most_frequent"), bloque="C").fit(X_tr, y_tr)


    X_te, y_te = cargar_prueba()
    print(f"\nPrueba: {len(y_te)} hogares (entrenamiento: {len(y_tr)})\n")
    assert not set(X_te.index) & set(X_tr.index), "hay hogares en entrenamiento y prueba a la vez"

    rng = np.random.default_rng(SEED)
    filas, errores, confusion = [], [], []
    pred = pd.DataFrame({"hogar_id": X_te.index, "ses_real": y_te.to_numpy()})
    predicciones_fig = {}

    candidatos = [("C", "ppom", finales["C"]), ("B", "ppom", finales["B"]),
                  ("ninguno", "baseline_mayoritaria", base)]
    for bloque, nombre, pipe in candidatos:
        y_hat = pipe.predict(X_te)
        valores = resumen_metricas(y_te, y_hat)
        inf, sup = bootstrap_ic(y_te, y_hat, rng)
        cv_media = media_cv(bloque) if nombre == "ppom" else {}
        for m, v in valores.items():
            filas.append({"bloque": bloque, "modelo": nombre, "metrica": m, "prueba": round(v, 3),
                          "ic95_inf": round(inf[m], 3), "ic95_sup": round(sup[m], 3),
                          "cv_media": cv_media.get(m, np.nan)})
        errores.append({"bloque": bloque, "modelo": nombre, **errores_graves(y_te, y_hat)})

        if nombre == "ppom":
            pred[f"pred_{bloque}"] = y_hat
            cm = confusion_matrix(y_te, y_hat, labels=CLASES)
            confusion.append(pd.DataFrame(cm, index=[NOMBRES_CLASES[c] for c in CLASES],
                                          columns=[NOMBRES_CLASES[c] for c in CLASES])
                             .rename_axis("real").reset_index().assign(bloque=bloque))
            predicciones_fig[ESCENARIOS[bloque]] = y_hat

    proba = finales["C"].predict_proba(X_te)
    for c in CLASES:
        pred[f"p_{NOMBRES_CLASES[c].lower()}_C"] = proba[:, c].round(4)
    cruces = {b: finales[b].named_steps["modelo"].tasa_cruces(
        finales[b].named_steps["prep"].transform(X_te)) for b in ESCENARIOS}


    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    MODELOS.mkdir(parents=True, exist_ok=True)

    metricas = pd.DataFrame(filas)
    metricas.to_csv(TABLAS / "test_metricas.csv", index=False)
    pd.DataFrame(errores).to_csv(TABLAS / "test_errores.csv", index=False)
    pd.concat(confusion, ignore_index=True).to_csv(TABLAS / "test_confusion.csv", index=False)
    pred.to_csv(TABLAS / "test_predicciones.csv", index=False)
    figura_confusion(y_te, predicciones_fig, FIGURAS / "test_confusion.png")
    for bloque, pipe in finales.items():
        joblib.dump(pipe, MODELOS / ARCHIVO_MODELO[bloque])

    SELLO.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": "PPOM", "escenario_final": "C (A+B+C)", "referencia": "B (A+B)",
        "alpha_gamma": hiper, "tasa_cruces_prueba": cruces,
        "n_entrenamiento": int(len(y_tr)), "n_prueba": int(len(y_te)),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 4. Resumen en la terminal ----
    ancho = metricas.pivot_table(index=["bloque", "modelo"], columns="metrica", values="prueba")
    print("Métricas en prueba (mae y errores lejanos: menor es mejor)")
    print(ancho.to_string())
    print("\nPrueba vs. validación cruzada, PPOM (IC 95 % por bootstrap)")
    print(metricas[metricas["modelo"] == "ppom"].to_string(index=False))
    print("\nErrores graves")
    print(pd.DataFrame(errores).to_string(index=False))
    print(f"\nTasa de cruces en prueba: {cruces}")
    print(f"\nModelos guardados en models/: {list(ARCHIVO_MODELO.values())}")
    print(f"Sello escrito en {SELLO.relative_to(ROOT)}: el conjunto de prueba queda usado.")


if __name__ == "__main__":
    main(forzar="--forzar" in sys.argv)