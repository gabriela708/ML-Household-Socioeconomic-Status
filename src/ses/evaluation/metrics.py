"""Métricas de evaluación para el SES ordinal del hogar.

Clases: 0 Pobre, 1 Vulnerable, 2 Media, 3 Alta.

Por qué estas métricas (ver sección 7.4 del informe):
- f1_macro             principal; da el mismo peso a las 4 clases (desbalance: Alta = 4,4 %)
                       NO distingue la gravedad de los errores ordinales (corrección 1.ª entrega).
- qwk                  kappa con pesos cuadráticos: castiga más los errores lejanos (orden).
- mae_ordinal          promedio de |real - predicho|, en "número de clases".
- pct_errores_lejanos  % de hogares a distancia >= 2 de su clase real (errores graves).
- recall_pobre         de los hogares pobres, cuántos detecta (costo social en focalización).
- recall_alta          clase minoritaria; se reporta como limitación.

Supuesto: qwk y mae_ordinal tratan las clases como igualmente espaciadas.

Uso:
    from ses.evaluation.metrics import SCORERS, resumen_metricas
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS)   # validación cruzada
    resumen_metricas(y_real, y_pred)                           # dict con todas, signo normal

IMPORTANTE:
scikit-learn siempre maximiza, así que en cross_validatemae_ordinal y pct_errores_lejanos salen NEGATIVOS (greater_is_better=False).
Para reportarlas, multiplicar por -1 (lo hace a_signo_normal()).
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, make_scorer, recall_score

CLASES = [0, 1, 2, 3]
NOMBRES_CLASES = {0: "Pobre", 1: "Vulnerable", 2: "Media", 3: "Alta"}
POBRE, ALTA = 0, 3

def macro_f1(y_true, y_pred):
    """F1 de cada clase, promediado sin ponderar por tamaño."""
    return f1_score(y_true, y_pred, labels=CLASES, average="macro", zero_division=0)

def qwk(y_true, y_pred):
    """Kappa de Cohen con pesos cuadráticos (error a distancia d pesa d²)."""
    return cohen_kappa_score(y_true, y_pred, labels=CLASES, weights="quadratic")

def mae_ordinal(y_true, y_pred):
    """Distancia promedio, en clases, entre la clase real y la predicha."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def pct_errores_lejanos(y_true, y_pred, distancia=2):
    """Proporción de hogares predichos a distancia o más clases de la real."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) >= distancia))

def recall_clase(y_true, y_pred, clase):
    """De los hogares que realmente son de clase, proporción que el modelo detecta."""
    return recall_score(y_true, y_pred, labels=[clase], average="macro", zero_division=0)

def recall_pobre(y_true, y_pred):
    return recall_clase(y_true, y_pred, POBRE)

def recall_alta(y_true, y_pred):
    return recall_clase(y_true, y_pred, ALTA)


MENOR_ES_MEJOR = ["mae_ordinal", "pct_errores_lejanos"]

SCORERS = {
    "f1_macro": make_scorer(macro_f1),
    "qwk": make_scorer(qwk),
    "mae_ordinal": make_scorer(mae_ordinal, greater_is_better=False),
    "pct_errores_lejanos": make_scorer(pct_errores_lejanos, greater_is_better=False),
    "recall_pobre": make_scorer(recall_pobre),
    "recall_alta": make_scorer(recall_alta),
}

METRICA_PRINCIPAL = "f1_macro"

# Para visualizar
def resumen_metricas(y_true, y_pred):
    """Todas las métricas con su signo normal (útil con cross_val_predict)."""
    return {
        "f1_macro": macro_f1(y_true, y_pred),
        "qwk": qwk(y_true, y_pred),
        "mae_ordinal": mae_ordinal(y_true, y_pred),
        "pct_errores_lejanos": pct_errores_lejanos(y_true, y_pred),
        "recall_pobre": recall_pobre(y_true, y_pred),
        "recall_alta": recall_alta(y_true, y_pred),
    }


def a_signo_normal(metrica, valor):
    """Revierte el signo negativo que cross_validate pone a las métricas 'menor es mejor'."""
    return -valor if metrica in MENOR_ES_MEJOR else valor