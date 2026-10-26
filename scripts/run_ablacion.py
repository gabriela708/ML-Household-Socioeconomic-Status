"""archivo lo cree para ver cuánto aporta cada bloque (A, B , C ) 
para poder hacer la ablacion de estas para ver de que varaibles depende el modelo. """

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

from ses.config import ROOT
from ses.pipeline import cargar_entrenamiento, construir_pipeline
X, y, cv = cargar_entrenamiento()

print(X.shape, y.value_counts())

# 2. Métricas
# PROVISIONAL (Gaby): solo macro-F1, que está en la lista de metrics.py.
# Cuando Gaby suba metrics.py, este diccionario se reemplaza por un import de ahí
# (QWK, MAE ordinal, % errores a distancia >= 2, recall de Pobre).
# ---------------------------------------------------------------------------
METRICAS = {
    "f1_macro": "f1_macro",
}

#Modelo provisional mientras silva sube el pom 
MODELOS = {
    "logistica": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    # PROVISIONAL (Silva): descomentar cuando suba ordinal.py
    # "ordinal_pom": lambda: ModeloOrdinal(),
}

BLOQUES = ["A", "B", "C"]
filas = []  # una fila por cada número, formato del contrato


def guardar(modelo, bloque, resultado):
    """Pasa los 5 puntajes de cada métrica a filas de la tabla."""
    for metrica in METRICAS:
        for fold, valor in enumerate(resultado[f"test_{metrica}"], start=1):
            filas.append({"modelo": modelo, "bloque": bloque,
                          "fold": fold, "metrica": metrica, "valor": valor})



r = cross_validate(DummyClassifier(), X, y, cv=cv, scoring=METRICAS)
guardar("dummy", "ninguno", r)
print(f"{'dummy':12s} línea base: f1_macro = {r['test_f1_macro'].mean():.3f}")
# Teniendo en cuenta el resultado que fue 0.16 se espera que cualquier bloque que de verdad aprenda algo tiene que superar 0,16.

#Ablación 
for nombre, crear_modelo in MODELOS.items():
    for bloque in BLOQUES:
        pipe = construir_pipeline(crear_modelo(), bloque)
        r = cross_validate(pipe, X, y, cv=cv, scoring=METRICAS)
        guardar(nombre, bloque, r)
        print(f"{nombre:12s} bloque {bloque}: f1_macro = {r['test_f1_macro'].mean():.3f}")


salida = ROOT / "reports" / "tables"
salida.mkdir(parents=True, exist_ok=True)

tabla = pd.DataFrame(filas)
tabla.to_csv(salida / "ablacion.csv", index=False)

resumen = (tabla.groupby(["modelo", "bloque", "metrica"])["valor"]
                .agg(media="mean", desv="std")
                .round(3)
                .reset_index())
resumen.to_csv(salida / "ablacion_resumen.csv", index=False)
print(resumen)




#Grafica
f1 = resumen[resumen["metrica"] == "f1_macro"]
base = f1.loc[f1["modelo"] == "dummy", "media"].iloc[0]
log = f1[f1["modelo"] == "logistica"]

fig, ax = plt.subplots(figsize=(6, 4))
etiquetas = ["A", "A+B", "A+B+C"]
ax.bar(etiquetas, log["media"], yerr=log["desv"], capsize=5, color="#4C72B0")
ax.axhline(base, color="gray", linestyle="--", label=f"Línea base ({base:.3f})")

for i, (m, d) in enumerate(zip(log["media"], log["desv"])):
    ax.text(i, m + d + 0.01, f"{m:.3f}", ha="center")

ax.set_ylabel("Macro-F1 (media de 5 folds)")
ax.set_xlabel("Bloques de variables")
ax.set_title("Ablación por bloques: regresión logística")
ax.set_ylim(0, 0.7)
ax.legend()

figuras = ROOT / "reports" / "figures"
figuras.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(figuras / "ablacion.png", dpi=150)