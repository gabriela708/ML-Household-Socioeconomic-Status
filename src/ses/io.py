"""Lectura de los módulos GEIH.

TODO:
- read_module(nombre): lee data/raw/<file> con sep=';', decimal=',' y la codificación de config.
- Convertir a numérico columnas leídas como texto (p. ej. FEX_C18).
- make_hogar_id(df): crea hogar_id.
- check_keys(df, nivel): verifica que la llave sea única.

Responsable: Silva. Día 1.
"""

import pandas as pd
from ses.config import CFG, ROOT


def read_module(name: str) -> pd.DataFrame:
    raise NotImplementedError


def make_hogar_id(df: pd.DataFrame) -> pd.Series:
    raise NotImplementedError
