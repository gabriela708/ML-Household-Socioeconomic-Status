"""Interpretación del PPOM final y contraste con el paper (Lelisho et al., 2022).

Con el 80 % de entrenamiento (el test no se toca), bloque C:
1. Odds ratios del PPOM con IC 95 % (máxima verosimilitud, sin pesos ni penalización:
   una fila por variable proporcional y una por corte en las relajadas.
2. Contraste con las variables significativas del paper: edad, educación y tamaño del
   hogar del jefe, y hábito de ahorro (aquí: cuenta de ahorros); sexo del jefe (no
   significativo en el paper).

Salidas:
    reports/tables/tabla7_ppom_or.csv        todos los OR del modelo final
    reports/tables/tabla8_contraste_paper.csv
    reports/figures/or_contraste_paper.png

Uso:
    python scripts/interpretacion_ppom.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from ses.config import ROOT
from ses.models.ordinal import preparar_X_inferencia, tabla_or_ppom
from ses.pipeline import cargar_entrenamiento

TABLAS = ROOT / "reports" / "tables"
FIGURAS = ROOT / "reports" / "figures"
AZUL, NARANJA, TINTA, TINTA_SUAVE, REJILLA = "#2a78d6", "#eb6834", "#1a1a19", "#52514e", "#e4e3df"


PAPER = {
    "Edad del jefe": ("num__edad_jefe", "Significativa (+); única que violó el supuesto"),
    "Educación del jefe": ("num__educ_jefe", "Significativa (+): secundaria o más → SES más alto"),
    "Tamaño del hogar": ("num__n_personas", "Significativa (−): más miembros → SES más bajo"),
    "Hábito de ahorro": ("bin__fin_cuenta_ahorros", "Significativa (+)"),
    "Sexo del jefe (mujer)": ("nom__sexo_jefe_2.0", "No significativa"),
}
NOMBRE_CORTE = {"todos": "todos los cortes", "corte 1": "≥ Vulnerable",
                "corte 2": "≥ Media", "corte 3": "Alta"}


def figura(contraste, ruta):
    d = contraste.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.2))
    for i, f in d.iterrows():
        color = NARANJA if f["corte"] != "todos" else AZUL
        ax.plot([f["OR_IC95_inf"], f["OR_IC95_sup"]], [i, i], color=color, lw=1.8)
        ax.scatter(f["OR"], i, s=45, color=color, edgecolor="white", lw=1.2, zorder=3)
        ax.annotate(f"{f['OR']:.2f}", (f["OR_IC95_sup"], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TINTA_SUAVE)
    ax.axvline(1, color=TINTA_SUAVE, lw=1, ls="--")
    ax.set_xscale("log")
    marcas = [v for v in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16)
              if d["OR_IC95_inf"].min() / 1.5 <= v <= d["OR_IC95_sup"].max() * 1.5]
    ax.set_xticks(marcas, [f"{v:g}" for v in marcas])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(d)), [f"{v} ({NOMBRE_CORTE[c]})" for v, c in zip(d["paper"], d["corte"])],
                  fontsize=8.5, color=TINTA)
    ax.set_xlabel("Odds ratio (escala log); > 1 = más probable una clase más alta",
                  fontsize=8.5, color=TINTA_SUAVE)
    ax.grid(axis="x", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
    leyenda = [plt.Line2D([], [], marker="o", color=AZUL, label="Efecto proporcional (igual en todos los cortes)"),
               plt.Line2D([], [], marker="o", color=NARANJA, label="Efecto relajado (distinto por corte)")]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.6 / len(d)),
              ncol=2, frameon=False, fontsize=8)
    ax.set_title("Variables del paper en el PPOM final (A+B+C), OR con IC 95 %",
                 fontsize=10.5, color=TINTA, loc="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    X, y, _ = cargar_entrenamiento()
    Xp = preparar_X_inferencia(X)
    print(f"Ajustando el PPOM con {len(y)} hogares y {Xp.shape[1]} columnas (puede tardar unos minutos)...")
    tabla = tabla_or_ppom(Xp, y.to_numpy())

    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(TABLAS / "tabla7_ppom_or.csv", index=False)

    filas = []
    for nombre, (col, hallazgo) in PAPER.items():
        sub = tabla[tabla["variable"] == col]
        if sub.empty:
            print(f"AVISO: {col} no está en el modelo")
            continue
        for _, f in sub.iterrows():
            filas.append({"paper": nombre, "variable": col, "corte": f["corte"],
                          "OR": f["OR"], "OR_IC95_inf": f["OR_IC95_inf"], "OR_IC95_sup": f["OR_IC95_sup"],
                          "p": f["p"], "hallazgo_paper": hallazgo})
    contraste = pd.DataFrame(filas)
    contraste.round(4).to_csv(TABLAS / "tabla8_contraste_paper.csv", index=False)
    figura(contraste, FIGURAS / "or_contraste_paper.png")

    relajadas = tabla[tabla["corte"] != "todos"]["variable"].unique()
    print(f"\nVariables relajadas ({len(relajadas)}): {list(relajadas)}")
    print("\nContraste con el paper (numéricas: OR por cada desviación estándar)")
    print(contraste.round(3).to_string(index=False))
    print("\nOR más grandes y más pequeños del modelo (p < 0,05)")
    sig = tabla[tabla["p"] < 0.05].sort_values("OR")
    print(pd.concat([sig.head(8), sig.tail(8)]).round(3).to_string(index=False))
    print("\nGuardado: tabla7_ppom_or.csv, tabla8_contraste_paper.csv y or_contraste_paper.png")


if __name__ == "__main__":
    main()