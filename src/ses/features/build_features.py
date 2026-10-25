"""Construcción de X a nivel hogar (jefe, hogar, vivienda), filtro de cabeceras y casos completos.

Responsable: Juanita. Día 2.

Entrada:  data/raw/ (módulos GEIH: hogar_vivienda, personas, ocupados)
Salida:   data/processed/X_hogar.parquet           (contrato de datos)
          data/interim/excluidos_hogar.parquet     (para docs/reduccion_muestra.md)
          data/interim/embudo.csv                  (para docs/reduccion_muestra.md)

Uso:      python src/ses/features/build_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas (funcionan en cualquier computador)
# ---------------------------------------------------------------------------
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists())
RAW = RAIZ / "data" / "raw"
INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"
sys.path.insert(0, str(RAIZ / "src"))

from ses.features.leakage import assert_no_leakage  # noqa: E402

ARCHIVO_HOGAR = "hogar_vivienda.csv"
ARCHIVO_PERSONAS = "personas_salud_educacion.csv"   # <- nombre exacto del archivo
ARCHIVO_OCUPADOS = "ocupados.csv"

# ---------------------------------------------------------------------------
# Variables (ver docs/matriz_fuga.md)
# ---------------------------------------------------------------------------
COLS_P5222 = [f"P5222S{i}" for i in range(1, 12)]   # productos financieros (selección múltiple)

COLS_VIVIENDA = [
    "P4005",                                                # tipo de vivienda
    "P4010", "P4020",                                       # material de paredes y pisos
    "P4030S1", "P4030S2", "P4030S3", "P4030S4", "P4030S5",  # servicios públicos
    "P4030S1A1",                                            # estrato
    "P5000", "P5010",                                       # cuartos y dormitorios
    "P5020",                                                # sanitario
    "P5070", "P5080",                                       # lugar y combustible para cocinar
    "P5090",                                                # tenencia de la vivienda
] + COLS_P5222

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

COLS_SERVICIOS = ["tiene_energia", "tiene_gas_natural", "tiene_alcantarillado",
                  "tiene_recoleccion_basuras", "tiene_acueducto"]


# ---------------------------------------------------------------------------
# Pasos 1-2: lectura y llave
# ---------------------------------------------------------------------------
def leer(nombre):
    """Lee un CSV de data/raw. Prueba utf-8 y, si falla por las tildes, latin-1."""
    try:
        return pd.read_csv(RAW / nombre, sep=";", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(RAW / nombre, sep=";", encoding="latin-1", low_memory=False)


def crear_llave(tabla):
    """Agrega hogar_id = DIRECTORIO_SECUENCIA_P_HOGAR (texto), como dice el contrato."""
    llave = (
        tabla["DIRECTORIO"].astype(str) + "_"
        + tabla["SECUENCIA_P"].astype(str) + "_"
        + tabla["HOGAR"].astype(str)
    )
    return pd.concat([tabla, llave.rename("hogar_id")], axis=1)


def cargar_datos():
    """Lee los tres módulos y les agrega la llave hogar_id."""
    hogar = crear_llave(leer(ARCHIVO_HOGAR))
    personas = crear_llave(leer(ARCHIVO_PERSONAS))
    ocupados = crear_llave(leer(ARCHIVO_OCUPADOS))
    assert hogar["hogar_id"].is_unique, "hogar_id repetido en la tabla de hogares"
    return hogar, personas, ocupados


# ---------------------------------------------------------------------------
# Pasos 3-6: variables por bloque
# ---------------------------------------------------------------------------
def variables_vivienda(hogar):
    """Paso 3: vivienda (bloque B), tamaño del hogar (bloque A) y CLASE para filtrar."""
    return hogar[["hogar_id", "CLASE", "P6008"] + COLS_VIVIENDA].copy()


def variables_jefe(personas):
    """Paso 4: edad, sexo y educación del jefe de hogar (P6050 = 1)."""
    jefe = personas.loc[personas["P6050"] == 1, ["hogar_id", "P6040", "P3271", "P3042"]].copy()
    assert jefe["hogar_id"].is_unique, "hay hogares con más de un jefe"
    return jefe


def resumen_personas(personas):
    """Paso 5: conteos y proporciones de las personas de cada hogar."""
    per = personas[["hogar_id", "P6040", "P6090", "P6100"]].copy()
    per["es_menor15"] = (per["P6040"] < 15).astype(int)
    per["es_mayor65"] = (per["P6040"] >= 65).astype(int)
    per["es_edad_trabajo"] = per["P6040"].between(15, 64).astype(int)
    per["afiliado_salud"] = per["P6090"].map({1: 1, 2: 0})   # 9 (no sabe) queda vacío
    per["subsidiado"] = (per["P6100"] == 3).astype(int)      # 3 = régimen subsidiado

    return per.groupby("hogar_id").agg(
        n_menores15=("es_menor15", "sum"),
        n_mayores65=("es_mayor65", "sum"),
        n_edad_trabajo=("es_edad_trabajo", "sum"),
        prop_afiliados_salud=("afiliado_salud", "mean"),
        prop_subsidiado=("subsidiado", "mean"),
    ).reset_index()


def resumen_ocupados(ocupados, personas):
    """Paso 6: ocupados y cotizantes del hogar, y posición ocupacional del jefe (bloque C)."""
    ocu = ocupados[["hogar_id", "ORDEN", "P6430", "P6920"]].copy()
    ocu["cotiza_pension"] = (ocu["P6920"] == 1).astype(int)   # 1 = sí cotiza

    resumen = ocu.groupby("hogar_id").agg(
        n_ocupados=("ORDEN", "count"),
        n_cotizantes=("cotiza_pension", "sum"),
    ).reset_index()

    orden_jefe = personas.loc[personas["P6050"] == 1, ["hogar_id", "ORDEN"]]
    pos_jefe = (ocu.merge(orden_jefe, on=["hogar_id", "ORDEN"])[["hogar_id", "P6430"]]
                .rename(columns={"P6430": "posicion_ocup_jefe"}))
    return resumen, pos_jefe


# ---------------------------------------------------------------------------
# Pasos 7-9: unión, limpieza y nombres
# ---------------------------------------------------------------------------
def unir(vivienda, jefe, res_personas, res_ocupados, pos_jefe):
    """Paso 7: una fila por hogar. Vacíos de ocupados = 0 (nadie trabaja)."""
    X = (vivienda
         .merge(jefe, on="hogar_id", how="left")
         .merge(res_personas, on="hogar_id", how="left")
         .merge(res_ocupados, on="hogar_id", how="left")
         .merge(pos_jefe, on="hogar_id", how="left"))
    assert len(X) == len(vivienda), "el merge duplicó hogares"

    X[["n_ocupados", "n_cotizantes"]] = X[["n_ocupados", "n_cotizantes"]].fillna(0)
    X["posicion_ocup_jefe"] = X["posicion_ocup_jefe"].fillna(0)   # 0 = jefe no ocupado
    return X


def limpiar_codigos(X):
    """Pasos 7b y 8: vacíos con significado y códigos de 'no sabe'."""
    # P5222 (selección múltiple): vacío = no marcó = 0.
    # Sin ninguna opción marcada o con "No sabe" (S10) -> vacío de verdad.
    sin_respuesta = X[COLS_P5222].isna().all(axis=1)
    no_sabe = X["P5222S10"] == 1
    X[COLS_P5222] = X[COLS_P5222].fillna(0)
    X.loc[sin_respuesta | no_sabe, COLS_P5222] = np.nan
    X = X.drop(columns=["P5222S10"])

    # P5080: no se pregunta si el hogar no cocina (P5070 = 6) -> 0 = "no cocina"
    X.loc[X["P5070"] == 6, "P5080"] = 0

    # Estrato: sin energía (no se preguntó) -> 7; 9 = no sabe -> vacío
    X.loc[X["P4030S1"] == 2, "P4030S1A1"] = 7
    X["P4030S1A1"] = X["P4030S1A1"].replace(9, np.nan)

    # Proporción de dependientes (reemplaza la razón de dependencia: nunca divide entre 0)
    X["prop_dependientes"] = (X["n_menores15"] + X["n_mayores65"]) / X["P6008"]
    return X


def renombrar(X):
    """Paso 9: nombres en snake_case y servicios como 1 = sí, 0 = no."""
    X = X.rename(columns=NOMBRES)
    X[COLS_SERVICIOS] = X[COLS_SERVICIOS].replace({2: 0})
    return X


# ---------------------------------------------------------------------------
# Paso 10: reducción de la muestra
# ---------------------------------------------------------------------------
def filtrar_muestra(X):
    """Cabeceras y casos completos. Devuelve X final, excluidos y el embudo."""
    embudo = {"Hogares GEIH": len(X)}

    X_cab = X[X["clase"] == 1].drop(columns=["clase"])
    embudo["Cabeceras (clase = 1)"] = len(X_cab)

    completo = X_cab.notna().all(axis=1)
    X_final = X_cab[completo].copy()
    excluidos = X_cab[~completo].copy()
    embudo["Casos completos"] = len(X_final)
    return X_final, excluidos, embudo


# ---------------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------------
def construir_X():
    """Corre los pasos 1 a 10 y devuelve X final, excluidos y embudo."""
    hogar, personas, ocupados = cargar_datos()
    res_ocupados, pos_jefe = resumen_ocupados(ocupados, personas)

    X = unir(
        variables_vivienda(hogar),
        variables_jefe(personas),
        resumen_personas(personas),
        res_ocupados,
        pos_jefe,
    )
    X = limpiar_codigos(X)
    X = renombrar(X)
    return filtrar_muestra(X)


def guardar(X_final, excluidos, embudo):
    """Paso 11: revisa la fuga y guarda los archivos."""
    assert_no_leakage(list(NOMBRES.keys()))                        # variables GEIH de origen
    assert_no_leakage(X_final.drop(columns=["hogar_id"]).columns)  # columnas finales

    PROCESSED.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    X_final.to_parquet(PROCESSED / "X_hogar.parquet", index=False)
    excluidos.to_parquet(INTERIM / "excluidos_hogar.parquet", index=False)
    pd.Series(embudo).to_csv(INTERIM / "embudo.csv", header=["hogares"])


if __name__ == "__main__":
    X_final, excluidos, embudo = construir_X()
    guardar(X_final, excluidos, embudo)

    for paso, n in embudo.items():
        print(f"{paso}: {n:,}")
    print("Excluidos por vacíos:", len(excluidos))
    print("Sin fuga ✔ — guardado data/processed/X_hogar.parquet", X_final.shape)