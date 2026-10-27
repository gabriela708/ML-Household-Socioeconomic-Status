"""Figuras y tabla de errores de la comparación de modelos (secciones 7.3 y 7.4).

Lee las tablas que escribe scripts/run_cv.py SIN REENTRENAR:
    reports/tables/cv_resumen.csv, predicciones_cv.csv, grilla_detalle.csv

Salidas:
    reports/figures/comparacion_modelos.png   6 métricas por modelo (media ± desv.)
    reports/figures/confusion_modelos.png     matrices de confusión (% de cada clase real)
    reports/figures/curva_knn.png             macro-F1 de KNN según k y la reducción con PCA
    reports/tables/errores_pobre.csv          errores Pobre <-> Media/Alta por modelo

Todo sale de validación cruzada (predicciones fuera de partición), no del conjunto de prueba.

Uso:
    python -m ses.evaluation.plots
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap 
from sklearn.metrics import confusion_matrix

from ses.config import ROOT
from ses.evaluation.metrics import CLASES, MENOR_ES_MEJOR, NOMBRES_CLASES

TABLAS = ROOT / "reports" / "tables"
FIGURAS = ROOT / "reports" / "figures"

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_SUAVE, REJILLA = "#1a1a19", "#52514e", "#e4e3df"
AZULES = LinearSegmentedColormap.from_list("azules", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

NOMBRE_MODELO = {
    "baseline_mayoritaria": "Línea base (mayoritaria)",
    "pom": "Logística ordinal (POM)",
    "lr_multinomial": "Logística multinomial",
    "lr_l2": "Logística L2 (Ridge)",
    "lr_l1": "Logística L1 (Lasso)",
    "naive_bayes": "Naive Bayes",
    "knn": "KNN",
}
TITULO_METRICA = {
    "f1_macro": "Macro-F1", "qwk": "QWK", "mae_ordinal": "MAE ordinal (clases)",
    "pct_errores_lejanos": "Errores a distancia ≥ 2", "recall_pobre": "Recall de Pobre",
    "recall_alta": "Recall de Alta",
}
BASE = "baseline_mayoritaria"

plt.rcParams.update({
    "font.size": 9, "text.color": TINTA, "axes.labelcolor": TINTA_SUAVE,
    "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE, "axes.edgecolor": REJILLA,
})


def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)


# 1. Comparación de modelos

def fig_comparacion_modelos(resumen, ruta):
    """Barras horizontales (media ± desv. en 5 folds); la línea base como línea punteada."""
    f1 = resumen[resumen["metrica"] == "f1_macro"].set_index("modelo")["media"]
    orden = f1.drop(BASE, errors="ignore").sort_values().index.tolist()  # mejor arriba
    metricas = list(TITULO_METRICA)

    fig, ejes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True)
    for ax, metrica in zip(ejes.flat, metricas):
        d = resumen[resumen["metrica"] == metrica].set_index("modelo")
        filas = d.loc[orden]
        colores = [NARANJA if m == "pom" else AZUL for m in orden]
        ax.barh(range(len(orden)), filas["media"], xerr=filas["desv"], height=0.62,
                color=colores, error_kw={"ecolor": TINTA_SUAVE, "capsize": 2, "linewidth": 0.8})
        if BASE in d.index:
            ax.axvline(d.loc[BASE, "media"], color=TINTA_SUAVE, linestyle="--", linewidth=1)
        if metrica == "f1_macro":  # etiquetas solo en la métrica principal
            for i, (v, s) in enumerate(zip(filas["media"], filas["desv"])):
                ax.text(v + s + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=TINTA)
        sentido = "↓ menor es mejor" if metrica in MENOR_ES_MEJOR else "↑ mayor es mejor"
        ax.set_title(f"{TITULO_METRICA[metrica]}\n({sentido})", fontsize=10)
        ax.set_yticks(range(len(orden)), [NOMBRE_MODELO.get(m, m) for m in orden])
        ax.set_xlim(left=0)
        _estilo(ax)

    leyenda = [plt.Rectangle((0, 0), 1, 1, color=NARANJA), plt.Rectangle((0, 0), 1, 1, color=AZUL),
               plt.Line2D([0], [0], color=TINTA_SUAVE, linestyle="--")]
    fig.legend(leyenda, ["Modelo principal (POM)", "Candidatos", "Línea base (mayoritaria)"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Comparación de modelos, bloque A+B+C (media ± desv. en 5 folds de validación cruzada)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# 2. Matrices de confusión (predicciones fuera de partición)

def matriz_porcentaje(y_real, y_pred):
    """Matriz de confusión normalizada por fila: % de cada clase real en cada predicha."""
    cm = confusion_matrix(y_real, y_pred, labels=CLASES)
    return 100 * cm / cm.sum(axis=1, keepdims=True)


def fig_matrices_confusion(pred, modelos, ruta):
    nombres = [NOMBRES_CLASES[c] for c in CLASES]
    columnas = 3
    filas_fig = int(np.ceil(len(modelos) / columnas))
    fig, ejes = plt.subplots(filas_fig, columnas, figsize=(4 * columnas, 3.7 * filas_fig))
    for ax in ejes.flat[len(modelos):]:
        ax.axis("off")
    for ax, modelo in zip(ejes.flat, modelos):
        cm = matriz_porcentaje(pred["ses_real"], pred[modelo])
        ax.imshow(cm, cmap=AZULES, vmin=0, vmax=100)
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                ax.text(j, i, f"{cm[i, j]:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if cm[i, j] > 55 else TINTA)
        ax.set_xticks(range(len(CLASES)), nombres, fontsize=8)
        ax.set_yticks(range(len(CLASES)), nombres, fontsize=8)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(NOMBRE_MODELO.get(modelo, modelo), fontsize=10)
    fig.suptitle("Matrices de Condusión con validación cruzada, bloque A+B+C)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# 3. Errores Pobre <-> Media/Alta (los más costosos para la focalización)

def tabla_errores_pobre(pred, modelos):
    """Por modelo: % de pobres predichos como Media/Alta y % de Media/Alta predichos como Pobre."""
    real = pred["ses_real"]
    pobres, media_alta = real == 0, real.isin([2, 3])
    filas = []
    for m in modelos:
        p = pred[m]
        filas.append({
            "modelo": m,
            "pobres": int(pobres.sum()),
            "pobre_a_media_alta_n": int((pobres & p.isin([2, 3])).sum()),
            "pobre_a_media_alta_pct": round(100 * (pobres & p.isin([2, 3])).sum() / pobres.sum(), 2),
            "media_alta": int(media_alta.sum()),
            "media_alta_a_pobre_n": int((media_alta & (p == 0)).sum()),
            "media_alta_a_pobre_pct": round(100 * (media_alta & (p == 0)).sum() / media_alta.sum(), 2),
        })
    return pd.DataFrame(filas)

# 4. Curva de validación de KNN

def _nombre_reduccion(texto):
    if texto == "passthrough":
        return "Sin PCA"
    return "PCA " + texto.split("=")[1].rstrip(")") + " componentes" if "=" in texto else texto


def fig_curva_knn(grilla, ruta):
    """Macro-F1 según k, una línea por opción de reducción, con la mejor distancia (p)."""
    knn = grilla[grilla["modelo"] == "knn"].copy()
    if knn.empty:
        return
    knn["k"] = knn["param_modelo__knn__n_neighbors"].astype(int)
    mejor_p = knn.loc[knn["f1_macro_media"].idxmax(), "param_modelo__knn__p"]
    knn = knn[knn["param_modelo__knn__p"] == mejor_p]
    distancia = {"1": "Manhattan", "2": "euclidiana"}.get(str(mejor_p), str(mejor_p))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for color, (opcion, d) in zip([AZUL, AQUA, "#eda100"], knn.groupby("param_modelo__reduccion", sort=False)):
        d = d.sort_values("k")
        ax.plot(d["k"], d["f1_macro_media"], color=color, linewidth=2, marker="o", markersize=6,
                label=_nombre_reduccion(opcion))
        ax.fill_between(d["k"], d["f1_macro_media"] - d["f1_macro_desv"],
                        d["f1_macro_media"] + d["f1_macro_desv"], color=color, alpha=0.12, linewidth=0)
    ax.set_xticks(sorted(knn["k"].unique()))
    ax.set_xlabel("Número de vecinos (k)")
    ax.set_ylabel("Macro-F1 (media de 5 folds)")
    ax.set_title(f"KNN: curva de validación (distancia {distancia})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=REJILLA, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)



def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    resumen = pd.read_csv(TABLAS / "cv_resumen.csv")
    pred = pd.read_csv(TABLAS / "predicciones_cv.csv")
    grilla = pd.read_csv(TABLAS / "grilla_detalle.csv", dtype=str)
    for c in grilla.columns:
        if c.endswith(("_media", "_desv")):
            grilla[c] = grilla[c].astype(float)

    modelos = [m for m in pred.columns if m not in ("hogar_id", "ses_real", BASE)]

    fig_comparacion_modelos(resumen, FIGURAS / "comparacion_modelos.png")
    fig_matrices_confusion(pred, modelos, FIGURAS / "confusion_modelos.png")
    fig_curva_knn(grilla, FIGURAS / "curva_knn.png")

    errores = tabla_errores_pobre(pred, [BASE] + modelos)
    errores.to_csv(TABLAS / "errores_pobre.csv", index=False)
    print(errores.to_string(index=False))
    print("\nGuardado: 3 figuras en reports/figures/ y errores_pobre.csv en reports/tables/")


if __name__ == "__main__":
    main()