"""Responsable: Juanita. Pruebas del Pipeline sin fuga (usa datos inventados)."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ses.features.leakage import assert_no_leakage
from ses.pipeline import BINARIAS, BLOQUES, NOMINALES, NUMERICAS, construir_pipeline


def datos_falsos(n=200):
    """Tabla con las 32 columnas y valores al azar del tipo correcto."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({c: rng.normal(40, 10, n) for c in NUMERICAS})
    for c in NOMINALES:
        X[c] = rng.choice([1, 2, 3], n)
    for c in BINARIAS:
        X[c] = rng.integers(0, 2, n)
    y = rng.integers(0, 4, n)
    return X, y


def test_tamano_de_los_bloques():
    assert len(BLOQUES["A"]) == 7
    assert len(BLOQUES["B"]) == 21
    assert len(BLOQUES["C"]) == 32
    assert len(set(BLOQUES["C"])) == 32          # sin repetidas


def test_ninguna_variable_prohibida():
    assert_no_leakage(BLOQUES["C"])


def test_escalador_aprende_solo_del_entrenamiento():
    X, y = datos_falsos()
    X.loc[100:, "edad_jefe"] = 1000              # la "validación" tiene edades absurdas
    pipe = construir_pipeline(DummyClassifier(), "A").fit(X.iloc[:100], y[:100])

    escalador = pipe.named_steps["prep"].named_transformers_["num"].named_steps["escalar"]
    pos = [c for c in NUMERICAS if c in BLOQUES["A"]].index("edad_jefe")
    assert escalador.mean_[pos] < 100            # si viera la validación, la media pasaría de 500


def test_categoria_nueva_no_rompe():
    X, y = datos_falsos()
    pipe = construir_pipeline(DummyClassifier(), "B").fit(X.iloc[:100], y[:100])
    validacion = X.iloc[100:].copy()
    validacion["material_paredes"] = 9           # código que el entrenamiento nunca vio
    assert len(pipe.predict(validacion)) == 100