"""Construcción de X a nivel hogar (jefe, hogar, vivienda), filtro de cabeceras y casos completos.

Responsable: Juanita. Día 2.
"""

# %%
import os
print("Estoy en:", os.getcwd())
print("¿Existe el archivo?", os.path.exists("data/raw/hogar_vivienda.csv"))


# %%
# Paso 0: ubicar la raíz del proyecto (funciona en cualquier computador)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists())
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
sys.path.insert(0, str(RAIZ / "src"))   # para poder importar leakage.py

from ses.features.leakage import assert_no_leakage

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
# Resumir los ocupados de cada hogar
ocu = ocupados[["hogar_id", "ORDEN", "P6430", "P6920"]].copy()
ocu["cotiza_pension"] = (ocu["P6920"] == 1).astype(int)          # 1 = sí cotiza

resumen_ocupados = ocu.groupby("hogar_id").agg(
    n_ocupados=("ORDEN", "count"),
    n_cotizantes=("cotiza_pension", "sum"),
).reset_index()

# Posición ocupacional del jefe: cruzar ocupados con el ORDEN del jefe
orden_jefe = personas.loc[personas["P6050"] == 1, ["hogar_id", "ORDEN"]]
pos_jefe = ocu.merge(orden_jefe, on=["hogar_id", "ORDEN"])[["hogar_id", "P6430"]]
pos_jefe = pos_jefe.rename(columns={"P6430": "posicion_ocup_jefe"})

print("resumen_ocupados:", resumen_ocupados.shape)
print("jefes ocupados:", pos_jefe.shape)

# %%
#unir todas las tablas en una sola fila por hogar
X = (
    vivienda
    .merge(jefe, on="hogar_id", how="left")
    .merge(resumen_personas, on="hogar_id", how="left")
    .merge(resumen_ocupados, on="hogar_id", how="left")
    .merge(pos_jefe, on="hogar_id", how="left")
)

# Hogares sin nadie trabajando: no aparecen en resumen_ocupados, así que quedan vacíos.
# En realidad tienen 0 ocupados y 0 cotizantes.
X[["n_ocupados", "n_cotizantes"]] = X[["n_ocupados", "n_cotizantes"]].fillna(0)

# Jefe que no trabaja: no tiene posición ocupacional. Se marca con 0 = "jefe no ocupado".
X["posicion_ocup_jefe"] = X["posicion_ocup_jefe"].fillna(0)

print("X:", X.shape)
print("¿un hogar por fila?", X["hogar_id"].is_unique)
X.head()

# %%
#vacíos que en realidad tienen significado

# P5222: pregunta de "marque todas". Vacío = no la marcó = 0
cols_p5222 = [f"P5222S{i}" for i in range(1, 12)]

# Antes de rellenar: ¿qué hogares no sabemos qué productos usan?
sin_respuesta = X[cols_p5222].isna().all(axis=1)    # no marcaron ninguna opción
no_sabe = X["P5222S10"] == 1                         # marcaron "No sabe"

X[cols_p5222] = X[cols_p5222].fillna(0)              # vacío = no marcó esa opción = 0
X.loc[sin_respuesta | no_sabe, cols_p5222] = np.nan  # esos hogares quedan vacíos de verdad
X = X.drop(columns=["P5222S10"])                     # "No sabe" ya no es un predictor

# P5080: vacío cuando el hogar no cocina (P5070 = 6). Se marca como 0 = "no cocina"
X.loc[X["P5070"] == 6, "P5080"] = 0

# Razón de dependencia -> proporción de dependientes (nunca divide entre 0)
X["prop_dependientes"] = (X["n_menores15"] + X["n_mayores65"]) / X["P6008"]
X = X.drop(columns=["razon_dependencia"])

X.isna().sum().sort_values(ascending=False).head(10)
# %%
# - vacío = el hogar no tiene energía (no se preguntó) -> categoría propia: 7 = "sin energía"
# - 9 = no sabe -> vacío de verdad (se elimina en el filtro de casos completos)
X.loc[X["P4030S1"] == 2, "P4030S1A1"] = 7
X["P4030S1A1"] = X["P4030S1A1"].replace(9, np.nan)

X["P4030S1A1"].value_counts(dropna=False).sort_index()
# %%
# Nombres de las variables 
NOMBRES = {
    "CLASE": "clase",
    "P6008": "n_personas",
    # Vivienda
    "P4005": "tipo_vivienda",
    "P4010": "material_paredes",
    "P4020": "material_pisos",
    "P4030S1": "tiene_energia",
    "P4030S2": "tiene_gas_natural",
    "P4030S3": "tiene_alcantarillado",
    "P4030S4": "tiene_recoleccion_basuras",
    "P4030S5": "tiene_acueducto",
    "P4030S1A1": "estrato",
    "P5000": "n_cuartos",
    "P5010": "n_dormitorios",
    "P5020": "tipo_sanitario",
    "P5070": "lugar_cocina",
    "P5080": "combustible_cocina",
    "P5090": "tenencia_vivienda",
    # Productos financieros (P5222)
    "P5222S1": "fin_cuenta_corriente",
    "P5222S2": "fin_cuenta_ahorros",
    "P5222S3": "fin_cdt",
    "P5222S4": "fin_prestamo_vivienda",
    "P5222S5": "fin_prestamo_vehiculo",
    "P5222S6": "fin_prestamo_libre_inversion",
    "P5222S7": "fin_tarjeta_credito",
    "P5222S8": "fin_otro",
    "P5222S9": "fin_ninguno",
    "P5222S11": "fin_billetera_movil",
    # Jefe de hogar
    "P6040": "edad_jefe",
    "P3271": "sexo_jefe",
    "P3042": "educ_jefe",
}

X = X.rename(columns=NOMBRES)
# Servicios: el DANE usa 1 = sí, 2 = no. Se pasa a 1 = sí, 0 = no
COLS_SERVICIOS = ["tiene_energia", "tiene_gas_natural", "tiene_alcantarillado",
                  "tiene_recoleccion_basuras", "tiene_acueducto"]
X[COLS_SERVICIOS] = X[COLS_SERVICIOS].replace({2: 0})
list(X.columns)
# %%
#verificar que no quede ninguna variable sin nombrar
[c for c in X.columns if c.startswith("P")]

# %%
# Paso 10: reducir la muestra y registrar el embudo
embudo = {"Hogares GEIH": len(X)}

# Filtro 1: solo cabeceras (zona urbana)
X_cab = X[X["clase"] == 1].drop(columns=["clase"])
embudo["Cabeceras (clase = 1)"] = len(X_cab)

# Filtro 2: casos completos (sin ningún vacío)
completo = X_cab.notna().all(axis=1)
X_final = X_cab[completo].copy()
excluidos = X_cab[~completo].copy()       # se guardan para compararlos en reduccion_muestra.md
embudo["Casos completos"] = len(X_final)

for paso, n in embudo.items():
    print(f"{paso}: {n:,}")
print("Excluidos por vacíos:", len(excluidos))

# ¿Qué variables causaron las exclusiones?
excluidos.isna().sum()[lambda s: s > 0]

# %%
# Revisar fuga 

# 1) Guardia de fuga
assert_no_leakage(list(NOMBRES.keys()))                        # variables GEIH de origen
assert_no_leakage(X_final.drop(columns=["hogar_id"]).columns)  # columnas finales (sin la llave)
print("Sin fuga ✔")

# 2) Archivo del contrato
PROCESSED.mkdir(parents=True, exist_ok=True)
X_final.to_parquet(PROCESSED / "X_hogar.parquet", index=False)
print("Guardado:", PROCESSED / "X_hogar.parquet", X_final.shape)

# 3) Material para docs/reduccion_muestra.md
INTERIM = RAIZ / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)
excluidos.to_parquet(INTERIM / "excluidos_hogar.parquet", index=False)
pd.Series(embudo).to_csv(INTERIM / "embudo.csv", header=["hogares"])
print("Guardados excluidos y embudo en data/interim")# %%
