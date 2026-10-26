"""ColumnTransformer + Pipeline (imputación, codificación, escalamiento) ajustado solo con entrenamiento.

Responsable: Juanita. Día 3.
"""
# %%
NUMERICAS = [
    "n_personas", "edad_jefe", "educ_jefe", "n_menores15", "n_mayores65", "prop_dependientes",
    "n_cuartos", "n_dormitorios",
    "prop_afiliados_salud", "prop_subsidiado", "n_ocupados", "n_cotizantes",
]
NOMINALES = [
    "sexo_jefe",
    "tipo_vivienda", "material_paredes", "material_pisos", "estrato", "tipo_sanitario",
    "lugar_cocina", "combustible_cocina", "tenencia_vivienda",
    "posicion_ocup_jefe",
]
BINARIAS = [
    "tiene_gas_natural", "tiene_alcantarillado", "tiene_recoleccion_basuras", "tiene_acueducto",
    "fin_corriente_o_cdt", "fin_cuenta_ahorros", "fin_credito_vivienda_vehiculo",
    "fin_prestamo_libre_inversion", "fin_tarjeta_credito", "fin_billetera_movil",
]


BLOQUE_A = ["n_personas", "edad_jefe", "sexo_jefe", "educ_jefe",
            "n_menores15", "n_mayores65", "prop_dependientes"]

BLOQUE_B = BLOQUE_A + [
    "tipo_vivienda", "material_paredes", "material_pisos",
    "tiene_gas_natural", "tiene_alcantarillado", "tiene_recoleccion_basuras", "tiene_acueducto",
    "estrato", "n_cuartos", "n_dormitorios", "tipo_sanitario",
    "lugar_cocina", "combustible_cocina", "tenencia_vivienda",
]

BLOQUE_C = BLOQUE_B + [
    "fin_corriente_o_cdt", "fin_cuenta_ahorros", "fin_credito_vivienda_vehiculo",
    "fin_prestamo_libre_inversion", "fin_tarjeta_credito", "fin_billetera_movil",
    "prop_afiliados_salud", "prop_subsidiado", "n_ocupados", "n_cotizantes",
    "posicion_ocup_jefe",
]

BLOQUES = {"A": BLOQUE_A, "B": BLOQUE_B, "C": BLOQUE_C}

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import pandas as pd
from ses.config import CFG, ROOT

def construir_preprocesador(columnas):
    """Imputa, codifica y escala solo las columnas del bloque."""
    num = [c for c in NUMERICAS if c in columnas]
    nom = [c for c in NOMINALES if c in columnas]
    bina = [c for c in BINARIAS if c in columnas]

    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())]), num),
        ("nom", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first",
                                                   sparse_output=False))]), nom),
        ("bin", SimpleImputer(strategy="most_frequent"), bina),
    ], remainder="drop")

def construir_pipeline(modelo, bloque="C"):
    """Preprocesamiento del bloque + modelo, en un solo objeto."""
    return Pipeline([
        ("prep", construir_preprocesador(BLOQUES[bloque])),
        ("modelo", modelo),
    ])


PROCESSED = ROOT / CFG["paths"]["processed"]

def cargar_entrenamiento():
    """X, y del 80 % y la lista de folds de split_indices.json."""
    split = json.loads((PROCESSED / "split_indices.json").read_text(encoding="utf-8"))
    ids = split["train_idx"]

    X = pd.read_parquet(PROCESSED / "X_hogar.parquet").set_index("hogar_id").loc[ids]
    y = pd.read_parquet(PROCESSED / "y_hogar.parquet").set_index("hogar_id")["ses"].loc[ids]

    posicion = pd.Series(range(len(ids)), index=ids)
    cv = []
    for fold in split["folds"]:
        val = posicion.loc[fold].to_numpy()
        entrena = posicion.drop(index=fold).to_numpy()
        cv.append((entrena, val))
    return X, y, cv
# %%
X, y, cv = cargar_entrenamiento()
print(X.shape)
print(y.value_counts(normalize=True).sort_index().round(3))
print([len(val) for _, val in cv])
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

base = cross_validate(DummyClassifier(), X, y, cv=cv, scoring="f1_macro")
print("Línea base:", base["test_score"].mean().round(3))

for b in ["A", "B", "C"]:
    modelo = LogisticRegression(max_iter=3000, class_weight="balanced")
    r = cross_validate(construir_pipeline(modelo, b), X, y, cv=cv, scoring="f1_macro")
    print(f"Bloque {b}: {r['test_score'].mean():.3f} ± {r['test_score'].std():.3f}")

"""
Gaby y silva para uqe tengan en cueta : 

    X, y, cv = cargar_entrenamiento()          # solo el 80 % + los 5 folds fijos
    pipe = construir_pipeline(modelo, bloque="C")   # "A", "B" o "C"
    cross_validate(pipe, X, y, cv=cv, scoring="f1_macro")

class_weight="balanced" va en el modelo, no aquí.
"""