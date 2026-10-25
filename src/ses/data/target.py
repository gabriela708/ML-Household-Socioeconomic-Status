"""Construcción del target: ingreso per cápita del hogar -> 4 clases DANE.

- load_pobreza(): lee la base de pobreza monetaria del DANE (data/external).
- load_hogares_cabecera(): hogares de la GEIH en cabecera (CLASE == 1) con su hogar_id.
- merge_ingreso(): cruza GEIH y DANE por DIRECTORIO + SECUENCIA_P (decisión #8).
- asignar_clase(): ypc + lp del hogar + umbrales de config -> ses (0..3).
- build_target(): todo lo anterior; devuelve las columnas del contrato de datos.

Regla (decisión #10: línea de pobreza `lp` de cada hogar, por dominio):
    0 Pobre       ypc <  lp
    1 Vulnerable  lp  <= ypc < vulnerable_max
    2 Media       vulnerable_max <= ypc < media_max
    3 Alta        ypc >= media_max

`ypc` (ingpcug) solo construye el target: nunca es predictor.
.
"""

import numpy as np
import pandas as pd

from ses.config import CFG, ROOT
from ses.io import read_module, make_hogar_id

POBREZA_FILE = "pobreza_monetaria_2025_hogares.csv"
COLUMNAS_CONTRATO = ["hogar_id", "DIRECTORIO", "ses", "ypc"]


def load_pobreza() -> pd.DataFrame:
    """Base de pobreza monetaria del DANE (diciembre), con nombres en mayúscula."""
    path = ROOT / CFG["paths"]["external"] / POBREZA_FILE
    df = pd.read_csv(path, usecols=["directorio", "secuencia_p", "clase",
                                    "ingpcug", "lp", "pobre"])
    df.columns = df.columns.str.upper()
    df = df[df["CLASE"] == CFG["poblacion"]["clase"]].drop(columns="CLASE")
    if df.duplicated(["DIRECTORIO", "SECUENCIA_P"]).any():
        raise ValueError("La base del DANE tiene hogares duplicados")
    return df


def load_hogares_cabecera() -> pd.DataFrame:
    """Hogares de la GEIH en cabecera, con hogar_id y las llaves para el cruce."""
    h = read_module("hogar_vivienda")
    h = h[h["CLASE"] == CFG["poblacion"]["clase"]].copy()
    h["hogar_id"] = make_hogar_id(h)
    return h[["hogar_id", "DIRECTORIO", "SECUENCIA_P"]]


def merge_ingreso(hogares: pd.DataFrame, pobreza: pd.DataFrame) -> pd.DataFrame:
    """Cruce interno: quedan solo los hogares de cabecera que están en la base del DANE."""
    df = hogares.merge(pobreza, on=["DIRECTORIO", "SECUENCIA_P"],
                       how="inner", validate="one_to_one")
    return df.rename(columns={"INGPCUG": "ypc", "LP": "lp", "POBRE": "pobre_dane"})


def asignar_clase(ypc: pd.Series, lp: pd.Series) -> pd.Series:
    """Clase DANE 0..3 a partir del ingreso per cápita y la línea de pobreza del hogar."""
    u = CFG["umbrales_clases_2025"]
    cond = [ypc < lp, ypc < u["vulnerable_max"], ypc < u["media_max"]]
    ses = np.select(cond, [0, 1, 2], default=3)
    return pd.Series(ses, index=ypc.index, name="ses").astype(int)


def build_target(con_auditoria: bool = False) -> pd.DataFrame:
    """y_hogar con las columnas del contrato.

    con_auditoria=True agrega `lp` y `pobre_dane` para verificar en el notebook
    (no se guardan en el parquet).
    """
    df = merge_ingreso(load_hogares_cabecera(), load_pobreza())
    if df["ypc"].isna().any() or df["lp"].isna().any():
        raise ValueError("Hay hogares sin ingreso o sin línea de pobreza")
    df["DIRECTORIO"] = df["DIRECTORIO"].astype("int64")
    df["ses"] = asignar_clase(df["ypc"], df["lp"])
    cols = COLUMNAS_CONTRATO + (["lp", "pobre_dane"] if con_auditoria else [])
    return df[cols].sort_values("hogar_id").reset_index(drop=True)
