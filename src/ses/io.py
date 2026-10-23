"""Lectura de los módulos GEIH.

- read_module(nombre): lee data/raw/<file> con sep=';', decimal=',' y la codificación de config.
- Convierte a numérico las columnas que llegan como texto (p. ej. FEX_C18, que viene con punto decimal).
- make_hogar_id(df): crea hogar_id = "DIRECTORIO_SECUENCIA_P_HOGAR".
- check_keys(df, nivel): verifica que la llave sea única.

Responsable: Silva. Día 1.
"""

import pandas as pd
from ses.config import CFG, ROOT

# Columnas que deben quedarse como texto (códigos con ceros a la izquierda o alfanuméricos).
TEXT_COLS = {"DPTO", "AREA", "REGIS"}


def _to_numeric(s: pd.Series) -> pd.Series:
    """Convierte texto a número aceptando coma o punto decimal; '.' y vacíos -> NaN."""
    s = s.astype("string").str.strip().replace({"": pd.NA, ".": pd.NA})
    out = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
    # Solo convertimos si no se pierde información (todo lo no vacío se pudo convertir).
    if out.notna().sum() == s.notna().sum():
        return out
    return s


def read_module(name: str) -> pd.DataFrame:
    """Lee un módulo GEIH por su nombre en config.yaml (p. ej. 'hogar_vivienda')."""
    if name not in CFG["raw_files"]:
        raise KeyError(f"Módulo desconocido: {name}. Opciones: {list(CFG['raw_files'])}")
    spec = CFG["raw_files"][name]
    path = ROOT / CFG["paths"]["raw"] / spec["file"]
    df = pd.read_csv(path, sep=";", decimal=",", encoding=spec["encoding"],
                     dtype=str, keep_default_na=True)
    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = _to_numeric(df[col])
    return df


def make_hogar_id(df: pd.DataFrame) -> pd.Series:
    """hogar_id = f"{DIRECTORIO}_{SECUENCIA_P}_{HOGAR}" (texto), según docs/contrato_datos.md."""
    k = CFG["keys"]["hogar"]
    parts = [df[c].astype("Int64").astype("string") for c in k]
    hid = parts[0].str.cat(parts[1:], sep="_")
    return hid.rename("hogar_id")


def check_keys(df: pd.DataFrame, nivel: str) -> bool:
    """True si la llave del nivel ('hogar' o 'persona') es única y sin faltantes."""
    k = CFG["keys"][nivel]
    if df[k].isna().any().any():
        raise ValueError(f"Hay faltantes en la llave {k}")
    dup = df.duplicated(subset=k).sum()
    if dup:
        raise ValueError(f"La llave {k} tiene {dup} duplicados")
    return True
