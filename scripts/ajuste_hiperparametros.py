"""Se crea este archivo para poder ajustar los parametros no 
se hace directamente en la ablacion ya que se pierde un orden
 pre establecido por el grupo y ademas para no mezclar rutas de 
 trabajo diferentes  """
# %% 

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
 
from ses.config import ROOT
from ses.pipeline import cargar_entrenamiento, construir_pipeline
 

X, y, cv = cargar_entrenamiento()
 


#    C es el INVERSO de la fuerza de regularización: C pequeño = modelo más simple.
#    Solo penalización L2 (la de por defecto); L1 queda en los candidatos de Gaby.

GRILLA_C = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {"modelo__C": GRILLA_C}      # "modelo" es el nombre del paso en construir_pipeline
 
BLOQUES = ["A", "B", "C"]
ETIQUETA_BLOQUE = {"A": "A", "B": "A+B", "C": "A+B+C"}
 
mejores = []   # mejor C de cada bloque → mejores_C.csv
detalle = []   # todos los C probados   → ajuste_C_detalle.csv

#  Búsqueda por bloque 
for bloque in BLOQUES:
    modelo = LogisticRegression(max_iter=3000, class_weight="balanced")   # decisión 33
    pipe = construir_pipeline(modelo, bloque)
 
    busqueda = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    busqueda.fit(X, y)
 
    res = busqueda.cv_results_
    i = busqueda.best_index_
    mejor_C = busqueda.best_params_["modelo__C"]
    mejores.append({"bloque": bloque, "mejor_C": mejor_C,
                    "f1_macro_media": res["mean_test_score"][i],
                    "f1_macro_desv": res["std_test_score"][i],
                    "en_borde_grilla": mejor_C in (GRILLA_C[0], GRILLA_C[-1])})
 
    for C, media, desv in zip(GRILLA_C, res["mean_test_score"], res["std_test_score"]):
        detalle.append({"bloque": bloque, "C": C, "f1_macro_media": media, "f1_macro_desv": desv})
 
    aviso = "  ← en el borde de la grilla: revisar" if mejor_C in (GRILLA_C[0], GRILLA_C[-1]) else ""
    print(f"Bloque {ETIQUETA_BLOQUE[bloque]:6s}: mejor C = {mejor_C:<6}  "
          f"f1_macro = {res['mean_test_score'][i]:.3f} ± {res['std_test_score'][i]:.3f}{aviso}")
#Tablas 
tablas = ROOT / "reports" / "tables"
figuras = ROOT / "reports" / "figures"
tablas.mkdir(parents=True, exist_ok=True)
figuras.mkdir(parents=True, exist_ok=True)
 
pd.DataFrame(mejores).round(4).to_csv(tablas / "mejores_C.csv", index=False)
detalle = pd.DataFrame(detalle).round(4)
detalle.to_csv(tablas / "ajuste_C_detalle.csv", index=False)
 

# Figura: macro-F1 según C (escala logarítmica), una línea por bloque

COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
fig, ax = plt.subplots(figsize=(7, 4.2))
for bloque in BLOQUES:
    d = detalle[detalle["bloque"] == bloque]
    ax.errorbar(d["C"], d["f1_macro_media"], yerr=d["f1_macro_desv"], marker="o",
                capsize=3, linewidth=2, color=COLOR[bloque], label=ETIQUETA_BLOQUE[bloque])
ax.set_xscale("log")
ax.set_ylim(0, 0.75)
ax.set_xlabel("C (escala logarítmica; mayor C = menos regularización)")
ax.set_ylabel("Macro-F1 (media de 5 folds)")
ax.set_title("Ajuste de C de la regresión logística por bloque")
ax.legend(title="Bloques", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(figuras / "ajuste_C.png", dpi=150)
plt.close(fig)
 
print("Guardado: mejores_C.csv, ajuste_C_detalle.csv y ajuste_C.png")
 
# %%
