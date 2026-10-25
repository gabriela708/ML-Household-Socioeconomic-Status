# %%
# Comparación de hogares incluidos vs excluidos (para docs/reduccion_muestra.md)
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists())
incluidos = pd.read_parquet(RAIZ / "data" / "processed" / "X_hogar.parquet")
excluidos = pd.read_parquet(RAIZ / "data" / "interim" / "excluidos_hogar.parquet")
print("Incluidos:", len(incluidos), "| Excluidos:", len(excluidos))

VARIABLES = ["n_personas", "edad_jefe", "educ_jefe", "n_cuartos", "n_menores15",
             "n_mayores65", "prop_dependientes", "n_ocupados", "n_cotizantes",
             "prop_subsidiado", "tiene_gas_natural", "tiene_alcantarillado"]

filas = []
for v in VARIABLES:
    a, b = incluidos[v], excluidos[v]
    sd = np.sqrt((a.var() + b.var()) / 2)
    filas.append({"variable": v,
                  "media_incluidos": round(a.mean(), 2),
                  "media_excluidos": round(b.mean(), 2),
                  "dif_estandarizada": round((b.mean() - a.mean()) / sd, 2)})

comparacion = pd.DataFrame(filas)
comparacion
# %%
