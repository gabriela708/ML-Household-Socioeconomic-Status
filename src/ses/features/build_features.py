"""Construcción de X a nivel hogar (jefe, hogar, vivienda), filtro de cabeceras y casos completos.

Responsable: Juanita. Día 2.
"""

# %%
import os
print("Estoy en:", os.getcwd())
print("¿Existe el archivo?", os.path.exists("data/raw/hogar_vivienda.csv"))

# %%
from pathlib import Path

# Busca hacia arriba la carpeta que contiene data/raw (la raíz del proyecto)
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists())
RAW = RAIZ / "data" / "raw"
print("Raíz del proyecto:", RAIZ)

# %%
import pandas as pd

def leer(nombre):
    """Lee un CSV de data/raw. Prueba primero utf-8 y, si falla por las tildes, latin-1."""
    try:
        return pd.read_csv(RAW / nombre, sep=";", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(RAW / nombre, sep=";", encoding="latin-1", low_memory=False)

hogar = leer("hogar_vivienda.csv")
personas = leer("personas_salud_educacion.csv")   # <- revisa el nombre exacto
ocupados = leer("ocupados.csv")

print("hogar:", hogar.shape)
print("personas:", personas.shape)
print("ocupados:", ocupados.shape)

# %%
def crear_llave(tabla):
    """Agrega hogar_id = DIRECTORIO_SECUENCIA_P_HOGAR (texto), como dice el contrato."""
    tabla["hogar_id"] = (
        tabla["DIRECTORIO"].astype(str) + "_"
        + tabla["SECUENCIA_P"].astype(str) + "_"
        + tabla["HOGAR"].astype(str)
    )
    return tabla

hogar = crear_llave(hogar)
personas = crear_llave(personas)
ocupados = crear_llave(ocupados)

print("¿hogar_id único en hogar?", hogar["hogar_id"].is_unique)
print(hogar["hogar_id"].head())



# %%
COLS_VIVIENDA = [
    "P4005",                                              # tipo de vivienda
    "P4010", "P4020",                                     # material de paredes y pisos
    "P4030S1", "P4030S2", "P4030S3", "P4030S4", "P4030S5",  # servicios públicos
    "P4030S1A1",                                          # estrato
    "P5000", "P5010",                                     # cuartos y dormitorios
    "P5020",                                              # sanitario
    "P5070", "P5080",                                     # lugar y combustible para cocinar
    "P5090",                                              # tenencia de la vivienda
] + [f"P5222S{i}" for i in range(1, 12)]                  # bienes del hogar (S1 a S11)

vivienda = hogar[["hogar_id", "CLASE", "P6008"] + COLS_VIVIENDA].copy()
print("vivienda:", vivienda.shape)

# %%
jefe = personas.loc[personas["P6050"] == 1, ["hogar_id", "P6040", "P3271", "P3042"]].copy()

print("jefe:", jefe.shape)
print("¿un jefe por hogar?", jefe["hogar_id"].is_unique)


# %%
# %%

import numpy as np

per = personas[["hogar_id", "P6040", "P6090", "P6100"]].copy()

# 1) Marcar a cada persona con 1/0
per["es_menor15"] = (per["P6040"] < 15).astype(int)
per["es_mayor65"] = (per["P6040"] >= 65).astype(int)
per["es_edad_trabajo"] = per["P6040"].between(15, 64).astype(int)
per["afiliado_salud"] = per["P6090"].map({1: 1, 2: 0})            # 9 (no sabe) queda vacío
per["subsidiado"] = (per["P6100"] == 3).astype(int)               # 3 = régimen subsidiado

# 2) Agrupar por hogar
resumen_personas = per.groupby("hogar_id").agg(
    n_menores15=("es_menor15", "sum"),
    n_mayores65=("es_mayor65", "sum"),
    n_edad_trabajo=("es_edad_trabajo", "sum"),
    prop_afiliados_salud=("afiliado_salud", "mean"),
    prop_subsidiado=("subsidiado", "mean"),
).reset_index()

# 3) Razón de dependencia: personas dependientes por cada persona en edad de trabajar
resumen_personas["razon_dependencia"] = (
    (resumen_personas["n_menores15"] + resumen_personas["n_mayores65"])
    / resumen_personas["n_edad_trabajo"].replace(0, np.nan)
)

print("resumen_personas:", resumen_personas.shape)
resumen_personas.head()

# %%
En personas un hogar de 4 personas tiene 4 filas. Para resumirlas se hacen dos cosas:

Marcar a cada persona con 1 o 0 según cumpla algo. Por ejemplo, es_menor15 vale 1 si tiene menos de 15 años.
Agrupar por hogar y sumar (cuántos cumplen) o promediar (qué proporción cumple).
