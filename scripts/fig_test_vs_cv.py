"""Figura: validación cruzada (entrenamiento) vs. conjunto de prueba, PPOM.

Lee reports/tables/test_metricas.csv (lo genera evaluar_test.py, que ya trae la media
de CV de cada métrica) y dibuja, por escenario, la cifra de CV y la de prueba con su
IC 95 % por bootstrap. No vuelve a abrir el conjunto de prueba.

Salida: reports/figures/test_vs_cv.png

Uso:
    python scripts/fig_test_vs_cv.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ses.config import ROOT

TABLAS = ROOT / "reports" / "tables"
FIGURAS = ROOT / "reports" / "figures"

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_SUAVE, REJILLA = "#1a1a19", "#52514e", "#e4e3df"

ESCENARIOS = {"C": "A+B+C (modelo final)", "B": "A+B (referencia)"}
MAYOR_MEJOR = {"f1_macro": "Macro-F1", "qwk": "QWK",
               "recall_pobre": "Recall de Pobre", "recall_alta": "Recall de Alta"}
MENOR_MEJOR = {"mae_ordinal": "MAE ordinal", "pct_errores_lejanos": "Errores a distancia ≥ 2"}


def panel(ax, datos, metricas, xlim, formato):
    """Una fila por métrica: punto de CV, punto de prueba con su IC y la línea que los une."""
    filas = list(metricas)[::-1]                       # la primera métrica arriba
    for i, m in enumerate(filas):
        d = datos.loc[m]

        ax.plot([d["cv_media"], d["prueba"]], [i + 0.13, i - 0.13], color=REJILLA, lw=1.5, zorder=1)
        ax.errorbar(d["prueba"], i - 0.13, xerr=[[d["prueba"] - d["ic95_inf"]], [d["ic95_sup"] - d["prueba"]]],
                    fmt="none", ecolor=NARANJA, elinewidth=1.5, capsize=3, zorder=2)
        ax.scatter(d["cv_media"], i + 0.13, s=60, color=AZUL, edgecolor="white", lw=1.5, zorder=3)
        ax.scatter(d["prueba"], i - 0.13, s=60, color=NARANJA, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(formato(d["prueba"]), (d["ic95_sup"], i - 0.13), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TINTA_SUAVE)
    ax.set_yticks(range(len(filas)), [metricas[m] for m in filas], fontsize=9, color=TINTA)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(filas) - 0.4)
    ax.grid(axis="x", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)
    ax.tick_params(axis="x", colors=TINTA_SUAVE, labelsize=8)
    ax.tick_params(axis="y", length=0)


def main():
    t = pd.read_csv(TABLAS / "test_metricas.csv")
    t = t[t["modelo"] == "ppom"]

    fig, ejes = plt.subplots(2, 2, figsize=(10, 5.6), height_ratios=[4, 2], sharex="row")
    for col, (bloque, titulo) in enumerate(ESCENARIOS.items()):
        datos = t[t["bloque"] == bloque].set_index("metrica")
        panel(ejes[0, col], datos, MAYOR_MEJOR, (0.4, 1.0), lambda v: f"{v:.3f}")
        panel(ejes[1, col], datos, MENOR_MEJOR, (0, 0.6), lambda v: f"{v:.3f}")
        ejes[0, col].set_title(titulo, fontsize=10.5, color=TINTA, loc="left", pad=10)
    for col in range(2):
        ejes[0, col].text(1, 1.03, "mayor es mejor", transform=ejes[0, col].transAxes,
                          ha="right", fontsize=8, color=TINTA_SUAVE)
        ejes[1, col].text(1, 1.05, "menor es mejor", transform=ejes[1, col].transAxes,
                          ha="right", fontsize=8, color=TINTA_SUAVE)

    leyenda = [plt.Line2D([], [], marker="o", ls="", color=AZUL, markersize=8,
                          label="Validación cruzada (5 folds, 80 %)"),
               plt.Line2D([], [], marker="o", ls="-", color=NARANJA, markersize=8,
                          label="Prueba (20 %) con IC 95 %")]
    fig.legend(handles=leyenda, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("PPOM: desempeño en validación cruzada vs. conjunto de prueba",
                 fontsize=12, color=TINTA, y=1.06)
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / "test_vs_cv.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Guardado: {ruta.relative_to(ROOT)}")


if __name__ == "__main__":
    main()