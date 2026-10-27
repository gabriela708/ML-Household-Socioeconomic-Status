"""Ablación por bloques: ¿cuánto aporta cada grupo de variables (A, B y C)?

Responsable: Juanita. Día 4.

Sirve para ver de qué variables depende el modelo. Compara A (demografía) →
A+B (+ vivienda y servicios) → A+B+C (+ empleo, salud y productos financieros)
contra la línea base, con los mismos 5 folds, las mismas métricas y el mismo
pipeline sin fuga para todos los modelos.

"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate
import warnings

# Categorías raras no vistas en un fold → se codifican como ceros (decisión 32).
warnings.filterwarnings("ignore", message="Found unknown categories")
from ses.config import ROOT
from ses.evaluation.metrics import CLASES, MENOR_ES_MEJOR, NOMBRES_CLASES, SCORERS, a_signo_normal
from ses.models.baseline import evaluar_baseline
from ses.pipeline import cargar_entrenamiento, construir_pipeline

# El POM de Silva (src/ses/models/ordinal.py). Si todavía no está en la rama o su
# archivo tiene un error, la ablación corre solo con la logística y AVISA el motivo.
try:
    from ses.models.ordinal import POM
    ERROR_POM = None
except (ImportError, SyntaxError) as error:
    POM, ERROR_POM = None, error


X, y, cv = cargar_entrenamiento()


# C de la logística por bloque: sale de ajuste_hiperparametros.py (decisión 34).
# Si todavía no se ha corrido el ajuste, usa C = 1 (el valor por defecto) y avisa.
archivo_C = ROOT / "reports" / "tables" / "mejores_C.csv"
if archivo_C.exists():
    C_POR_BLOQUE = pd.read_csv(archivo_C).set_index("bloque")["mejor_C"].to_dict()
else:
    C_POR_BLOQUE = {}
    print("AVISO: no existe mejores_C.csv; la logística usa C = 1. Corre antes ajuste_hiperparametros.py")


# C de la logística por bloque: sale de ajuste_hiperparametros.py (decisión 34).
# Si todavía no se ha corrido el ajuste, usa C = 1 (el valor por defecto) y avisa.
archivo_C = ROOT / "reports" / "tables" / "mejores_C.csv"
if archivo_C.exists():
    C_POR_BLOQUE = pd.read_csv(archivo_C).set_index("bloque")["mejor_C"].to_dict()
else:
    C_POR_BLOQUE = {}
    print("AVISO: no existe mejores_C.csv; la logística usa C = 1. Corre antes ajuste_hiperparametros.py")

# Los dos modelos usan class_weight="balanced" (decisiones 31 y 33).
MODELOS = {
    "logistica": lambda bloque: LogisticRegression(C=C_POR_BLOQUE.get(bloque, 1.0),
                                                   max_iter=3000, class_weight="balanced"),
}
if POM is not None:
    MODELOS["pom"] = lambda bloque: POM(class_weight="balanced")

# Modelo principal del proyecto (el del paper): el POM si está; si no, la logística
PRINCIPAL = "pom" if "pom" in MODELOS else "logistica"

BLOQUES = ["A", "B", "C"]
ETIQUETA_BLOQUE = {"A": "A", "B": "A+B", "C": "A+B+C"}       # los bloques son acumulativos
NOMBRE_MODELO = {"logistica": "Logística multinomial", "pom": "Logística ordinal (POM)"}


filas = []  # una fila por cada número, formato del contrato


def guardar(modelo, bloque, resultado):
    """Pasa los 5 puntajes de cada métrica a filas, con su signo normal."""
    for metrica in SCORERS:
        for fold, valor in enumerate(resultado[f"test_{metrica}"], start=1):
            filas.append({"modelo": modelo, "bloque": bloque, "fold": fold,
                          "metrica": metrica, "valor": a_signo_normal(metrica, valor)})


# Línea base del equipo (Gaby): clase mayoritaria, mismos folds y mismas métricas
base_df = evaluar_baseline(X, y, cv)
f1_base = base_df.loc[base_df["metrica"] == "f1_macro", "valor"].mean()
print(f"{'baseline':10s} línea base: f1_macro = {f1_base:.3f}")

for nombre, crear_modelo in MODELOS.items():
    for bloque in BLOQUES:
        pipe = construir_pipeline(crear_modelo(bloque), bloque)
        r = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS)
        guardar(nombre, bloque, r)
        print(f"{nombre:10s} bloque {ETIQUETA_BLOQUE[bloque]:6s}: "
              f"f1_macro = {r['test_f1_macro'].mean():.3f}")

#Tablas 
tablas = ROOT / "reports" / "tables"
figuras = ROOT / "reports" / "figures"
tablas.mkdir(parents=True, exist_ok=True)
figuras.mkdir(parents=True, exist_ok=True)

tabla = pd.concat([base_df, pd.DataFrame(filas)], ignore_index=True)
tabla.to_csv(tablas / "ablacion.csv", index=False)

resumen = (tabla.groupby(["modelo", "bloque", "metrica"])["valor"]
                .agg(media="mean", desv="std")
                .round(3)
                .reset_index())
resumen.to_csv(tablas / "ablacion_resumen.csv", index=False)

# Vista ancha para leer en la terminal: una fila por modelo y bloque, una columna por métrica
print(resumen.pivot_table(index=["modelo", "bloque"], columns="metrica", values="media")
             [list(SCORERS)].to_string())


# Gráficas para el informe

COLOR = {"logistica": "#2a78d6", "pom": "#eb6834"}
GRIS = "#52514e"
modelos = list(MODELOS)
x = np.arange(len(BLOQUES))
ancho = 0.8 / len(modelos)


def barras(ax, metrica, etiquetas_valor=False):
    """Barras por bloque (una por modelo), con desviación estándar y línea base."""
    for i, m in enumerate(modelos):
        d = (resumen[(resumen["modelo"] == m) & (resumen["metrica"] == metrica)]
             .set_index("bloque").loc[BLOQUES])
        pos = x + (i - (len(modelos) - 1) / 2) * ancho
        ax.bar(pos, d["media"], ancho * 0.92, yerr=d["desv"], capsize=3,
               color=COLOR[m], label=NOMBRE_MODELO[m])
        if etiquetas_valor:
            for p, v, s in zip(pos, d["media"], d["desv"]):
                ax.text(p, v + s + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    base = resumen.loc[(resumen["modelo"] == "baseline_mayoritaria")
                       & (resumen["metrica"] == metrica), "media"]
    if len(base):
        ax.axhline(base.iloc[0], color=GRIS, linestyle="--", linewidth=1,
                   label=f"Línea base ({base.iloc[0]:.3f})")
    ax.set_xticks(x, [ETIQUETA_BLOQUE[b] for b in BLOQUES])
    ax.spines[["top", "right"]].set_visible(False)


#  Figura principal: macro-F1 por bloque
fig, ax = plt.subplots(figsize=(7, 4.2))
barras(ax, "f1_macro", etiquetas_valor=True)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Macro-F1 (media de 5 folds)")
ax.set_xlabel("Bloques de variables (acumulativos)")
ax.set_title("Ablación por bloques: macro-F1")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(figuras / "ablacion.png", dpi=150)
plt.close(fig)

#  Las 6 métricas del equipo, un panel por métrica (cada una con su propia escala)
TITULO = {
    "f1_macro": "Macro-F1", "qwk": "QWK", "mae_ordinal": "MAE ordinal (clases)",
    "pct_errores_lejanos": "Errores a distancia ≥ 2", "recall_pobre": "Recall de Pobre",
    "recall_alta": "Recall de Alta",
}
fig, ejes = plt.subplots(2, 3, figsize=(11, 6.5))
for ax, metrica in zip(ejes.flat, SCORERS):
    barras(ax, metrica)
    sentido = "↓ menor es mejor" if metrica in MENOR_ES_MEJOR else "↑ mayor es mejor"
    ax.set_title(f"{TITULO.get(metrica, metrica)}\n({sentido})", fontsize=10)
    ax.set_ylim(bottom=0)
manejadores, nombres = ejes.flat[0].get_legend_handles_labels()
fig.legend(manejadores, nombres, loc="lower center", ncol=len(nombres), frameon=False, fontsize=9)
fig.suptitle("Ablación por bloques: métricas del equipo (media ± desv. en 5 folds)")
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(figuras / "ablacion_metricas.png", dpi=150)
plt.close(fig)

# Matrices de confusión del modelo principal: ¿dónde se equivoca?
#     cross_val_predict da, para cada hogar, la predicción del fold en que fue validación.
#     Se normaliza por fila: % de cada clase real que cae en cada clase predicha.
nombres_clases = [NOMBRES_CLASES[c] for c in CLASES]
fig, ejes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, bloque in zip(ejes, ["B", "C"]):
    pred = cross_val_predict(construir_pipeline(MODELOS[PRINCIPAL](bloque), bloque), X, y, cv=cv)
    cm = confusion_matrix(y, pred, labels=CLASES, normalize="true") * 100
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, f"{cm[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 55 else "black")
    ax.set_xticks(range(len(CLASES)), nombres_clases)
    ax.set_yticks(range(len(CLASES)), nombres_clases)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"{NOMBRE_MODELO[PRINCIPAL]}: bloques {ETIQUETA_BLOQUE[bloque]}", fontsize=10)
fig.suptitle("¿Dónde se equivoca el modelo? (% de cada clase real, 5 folds)")
fig.tight_layout()
fig.savefig(figuras / "ablacion_confusion.png", dpi=150)
plt.close(fig)

print("Guardado: ablacion.csv, ablacion_resumen.csv y 3 figuras en reports/")
# %%