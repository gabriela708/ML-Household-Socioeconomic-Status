"""Responsable: Gaby. Pruebas del PPOM (usa datos inventados, no lee la GEIH)."""
import numpy as np
from scipy.optimize import check_grad

from ses.models.ordinal import POM, PPOM


def datos_falsos(n=2000, seed=0):
    """4 clases ordenadas; la variable 0 NO cumple odds proporcionales en el último corte."""
    rng = np.random.default_rng(seed)
    X = np.c_[rng.normal(size=(n, 4)), rng.integers(0, 2, (n, 2))]
    lat = X @ np.array([1.0, 0.5, -0.5, 0.0, 0.8, 0.0])
    y = np.zeros(n, dtype=int)
    for j, c in enumerate([-1.0, 0.5, 2.5]):
        extra = 0.8 * X[:, 0] if j == 2 else 0
        y += (lat + extra + rng.logistic(size=n) > c)
    return X, y


def test_probabilidades_validas():
    X, y = datos_falsos()
    P = PPOM(class_weight="balanced").fit(X, y).predict_proba(X)
    assert P.shape == (len(y), 4)
    assert (P >= 0).all() and np.allclose(P.sum(axis=1), 1)


def test_autofit_detecta_la_variable_no_proporcional():
    X, y = datos_falsos()
    m = PPOM().fit(X, y)
    assert m.relajadas_[0]
    assert m.n_relajadas_ < X.shape[1]


def test_sin_variables_relajadas_es_el_pom():
    X, y = datos_falsos()
    ppom = PPOM(z=1e9).fit(X, y)
    pom = POM().fit(X, y)
    assert ppom.n_relajadas_ == 0
    assert np.abs(ppom.predict_proba(X) - pom.predict_proba(X)).max() < 1e-3


def test_gradiente_correcto():
    X, y = datos_falsos(n=500)
    m = PPOM(z=0).fit(X, y)
    Z = X[:, m.relajadas_]
    k = np.searchsorted(m.classes_, y)
    w = np.random.default_rng(1).uniform(0.5, 2, len(y))
    r = Z.shape[1]
    x = np.r_[np.full(X.shape[1], 0.1), np.full(2 * r, 0.02), -1.0, 0.4, 0.6]
    f = lambda p: m._perdida(p, X, Z, k, w)[0]
    g = lambda p: m._perdida(p, X, Z, k, w)[1]
    assert check_grad(f, g, x) / np.linalg.norm(g(x)) < 1e-4
